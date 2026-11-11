"""The docker run command, and real runs in a small image when a docker daemon is there."""

import subprocess
from pathlib import Path

import pytest

from conftest import docker_available
from remark.docker import docker_run_prefix, execute_docker, without_tag, THREAD_VARIABLES

IMAGE = 'bash:5.2'  # small, and has no conda: the script runs directly


def test_no_terminal_and_a_name(tmp_path):
    prefix = docker_run_prefix(tmp_path, 'img', workdir='/w', name='n')
    assert '-it' not in prefix and '-t' not in prefix and '-i' not in prefix
    # named and not --rm: its end is read from docker before execute_docker removes it
    assert prefix[prefix.index('--name') + 1] == 'n' and '--rm' not in prefix and prefix[-1] == 'img'
    assert f'type=bind,source={tmp_path.resolve()},target=/w' in prefix
    assert prefix[prefix.index('--workdir') + 1] == '/w'


def test_a_name_without_its_tag():
    assert without_tag('econ-ark/thing:abc') == without_tag('econ-ark/thing') == 'econ-ark/thing'
    assert without_tag('localhost:5000/econ-ark/thing') == 'localhost:5000/econ-ark/thing'
    assert without_tag('localhost:5000/econ-ark/thing:abc') == 'localhost:5000/econ-ark/thing'


def test_limits_only_when_given(tmp_path):
    bare = docker_run_prefix(tmp_path, 'img', workdir='/w', name='n')
    for flag in ('--network', '--cpus', '--memory', '--env'):
        assert flag not in bare
    capped = docker_run_prefix(tmp_path, 'img', workdir='/w', name='n', network=False, cpus=2,
                               memory='8g', threads=3)
    assert capped[capped.index('--network') + 1] == 'none'
    assert capped[capped.index('--cpus') + 1] == '2'
    assert capped[capped.index('--memory') + 1] == '8g'
    for variable in THREAD_VARIABLES:
        assert f'{variable}=3' in capped


needs_docker = pytest.mark.skipif(not docker_available(), reason='no working docker daemon')


@pytest.fixture(scope='module')
def image():
    subprocess.run(['docker', 'pull', '--quiet', IMAGE], check=True, capture_output=True)
    return IMAGE


def script(tmp_path, body):
    (tmp_path / 'run.sh').write_text(body)
    return tmp_path


def containers(tmp_path):
    """The containers execute_docker made for this checkout that docker still has, running or not."""
    names = subprocess.run(['docker', 'ps', '--all', '--format', '{{.Names}}'], capture_output=True, text=True).stdout
    return [n for n in names.split() if n.startswith(f'remark-{Path(tmp_path).name.lower()}-')]


@pytest.mark.docker
@needs_docker
def test_runs_without_a_terminal_in_the_checkout(tmp_path, image):
    repo = script(tmp_path, 'pwd; ls; echo "OMP=${OMP_NUM_THREADS:-unset}"; echo out > made.txt\n')
    proc = execute_docker(repo, image, 'run.sh', threads=2)
    assert proc.returncode == 0, proc.stdout
    assert '/remark' in proc.stdout and 'run.sh' in proc.stdout and 'OMP=2' in proc.stdout
    assert (repo / 'made.txt').read_text() == 'out\n'  # the checkout is writable
    assert proc.termination == 'exited' and proc.duration_seconds > 0 and proc.started <= proc.ended
    assert containers(tmp_path) == []  # removed when done


@pytest.mark.docker
@needs_docker
def test_exit_code_passes_through(tmp_path, image):
    proc = execute_docker(script(tmp_path, 'exit 7\n'), image, 'run.sh')
    assert (proc.returncode, proc.termination, proc.signal) == (7, 'exited', None)


@pytest.mark.docker
@needs_docker
def test_arguments_reach_the_script(tmp_path, image):
    proc = execute_docker(script(tmp_path, 'echo "$# $1 $2"\n'), image, 'run.sh', arguments=['--level', 'two words'])
    assert proc.stdout.strip() == '2 --level two words'


@pytest.mark.docker
@needs_docker
def test_a_signal_is_named(tmp_path, image):
    proc = execute_docker(script(tmp_path, 'kill -TERM $$\n'), image, 'run.sh')
    assert (proc.returncode, proc.termination, proc.signal) == (143, 'signal', 'SIGTERM')


@pytest.mark.docker
@needs_docker
def test_the_memory_limit_is_told_from_another_kill(tmp_path, image):
    # tail holds its whole input: 256 MB against a 32 MB limit
    repo = script(tmp_path, 'head -c 256m /dev/zero | tail > /dev/null\necho "tail ended $?"\nexit 137\n')
    # docker does not report every one (1 in 40 went unreported on docker 29.1.3), and an
    # unreported one reads as a kill: so up to five runs, of which one must be told apart.
    seen = []
    while len(seen) < 5 and 'oom' not in seen:
        proc = execute_docker(repo, image, 'run.sh', memory='32m')
        assert proc.returncode == 137 and (proc.termination, proc.signal) in (('oom', None), ('signal', 'SIGKILL'))
        seen.append(proc.termination)
    assert 'oom' in seen
    # the same exit code with no memory limit hit is a kill of another kind
    plain = execute_docker(script(tmp_path, 'exit 137\n'), image, 'run.sh', memory='32m')
    assert (plain.returncode, plain.termination, plain.signal) == (137, 'signal', 'SIGKILL')


@pytest.mark.docker
@needs_docker
def test_no_network(tmp_path, image):
    repo = script(tmp_path, 'ls /sys/class/net\n')
    assert execute_docker(repo, image, 'run.sh', network=False).stdout.split() == ['lo']
    assert len(execute_docker(repo, image, 'run.sh').stdout.split()) > 1


@pytest.mark.docker
@needs_docker
def test_timeout_kills_the_container(tmp_path, image):
    proc = execute_docker(script(tmp_path, 'echo started; sleep 60\n'), image, 'run.sh', timeout=3)
    assert (proc.returncode, proc.termination) == (124, 'timeout')
    assert 'started' in proc.stdout and 'TIMEOUT' in proc.stdout
    assert containers(tmp_path) == []

"""The docker run command, and real runs in a small image when a docker daemon is there."""

import subprocess
from pathlib import Path

import pytest

from conftest import docker_available
from remark.docker import docker_run_prefix, execute_docker, THREAD_VARIABLES

IMAGE = 'bash:5.2'  # small, and has no conda: the script runs directly


def test_no_terminal_and_removed_when_done(tmp_path):
    prefix = docker_run_prefix(tmp_path, 'img', workdir='/w', name='n')
    assert '-it' not in prefix and '-t' not in prefix and '-i' not in prefix
    assert '--rm' in prefix and prefix[-1] == 'img'
    assert f'type=bind,source={tmp_path.resolve()},target=/w' in prefix
    assert prefix[prefix.index('--workdir') + 1] == '/w'


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


@pytest.mark.docker
@needs_docker
def test_runs_without_a_terminal_in_the_checkout(tmp_path, image):
    repo = script(tmp_path, 'pwd; ls; echo "OMP=${OMP_NUM_THREADS:-unset}"; echo out > made.txt\n')
    proc = execute_docker(repo, image, 'run.sh', threads=2)
    assert proc.returncode == 0, proc.stdout
    assert '/remark' in proc.stdout and 'run.sh' in proc.stdout and 'OMP=2' in proc.stdout
    assert (repo / 'made.txt').read_text() == 'out\n'  # the checkout is writable


@pytest.mark.docker
@needs_docker
def test_exit_code_passes_through(tmp_path, image):
    proc = execute_docker(script(tmp_path, 'exit 7\n'), image, 'run.sh')
    assert proc.returncode == 7


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
    assert proc.returncode == 124
    assert 'TIMEOUT' in proc.stdout
    names = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], capture_output=True, text=True).stdout
    assert f'remark-{Path(tmp_path).name.lower()}-' not in names

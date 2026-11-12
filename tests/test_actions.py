"""build and execute with their records: real repo2docker builds of a small image, and
the refusals that keep a run from using an image built from something else."""

import json
import os
import re
import subprocess
from uuid import uuid4

import pytest

from conftest import git, repo2docker_available
from remark import records
from remark.actions import build, execute, environment_path, environment_spec, image_name
from remark.records import RecordError
from remark.cli import main
from remark.docker import clean_docker, image_id, image_label, COMMIT_LABEL, ENVIRONMENT_LABEL

REMOTE = 'https://example.org/test/thing'
IMAGE_ID = re.compile(r'^sha256:[0-9a-f]{64}$')

needs_repo2docker = pytest.mark.skipif(not repo2docker_available(), reason='no working docker daemon, or no repo2docker')


def commit_all(repo, message):
    git('add', '-A', cwd=repo)
    git('commit', '--quiet', '-m', message, cwd=repo)
    return git('rev-parse', 'HEAD', cwd=repo)


def make_repository(repo):
    """A small REMARK: binder/Dockerfile on bash, and a reproduce.sh that writes into the checkout."""
    (repo / 'binder').mkdir(parents=True)
    git('init', '--quiet', '--initial-branch', 'main', cwd=repo)
    git('config', 'user.email', 'test@example.org', cwd=repo)
    git('config', 'user.name', 'test', cwd=repo)
    git('remote', 'add', 'origin', REMOTE + '.git', cwd=repo)
    (repo / 'binder' / 'Dockerfile').write_text('FROM bash:5.2\n')
    (repo / 'reproduce.sh').write_text('echo "ran with OMP=${OMP_NUM_THREADS:-unset}"\ndate > made.txt\n')
    commit_all(repo, 'first')
    return repo


@pytest.fixture
def checkout(tmp_path):
    return make_repository(tmp_path / 'thing')


@pytest.fixture
def name():
    """An image name of this test's own, removed with every tag afterwards."""
    name = f'remark-test/{uuid4().hex[:12]}'
    yield name
    clean_docker(name)


def test_environment_directory(tmp_path):
    assert environment_path(tmp_path) == '.'
    (tmp_path / '.binder').mkdir()
    assert environment_path(tmp_path) == '.binder'
    (tmp_path / 'binder').mkdir()
    assert environment_path(tmp_path) == 'binder'
    assert image_name('econ-ark/thing', {'sha256': 'abcdef0123456789'}) == 'econ-ark/thing:abcdef012345'


def test_an_environment_in_a_directory_of_its_own(tmp_path):
    (tmp_path / 'binder').mkdir()
    (tmp_path / 'envs' / 'Solver One').mkdir(parents=True)
    (tmp_path / 'envs' / 'files').mkdir()
    (tmp_path / 'envs' / 'Solver One' / 'Dockerfile').write_text('FROM scratch\n')
    # the repository's own, however it is named: repo2docker on the whole checkout
    for own in (None, '.', 'binder', 'binder/'):
        assert environment_spec(tmp_path, own) == ('binder', 'repo2docker', True)
    # a directory of its own: its Dockerfile when it has one, else its binder files
    assert environment_spec(tmp_path, 'envs/Solver One/') == ('envs/Solver One', 'dockerfile', False)
    assert environment_spec(tmp_path, 'envs/files') == ('envs/files', 'repo2docker', False)
    for outside in ('envs/none', '../elsewhere', str(tmp_path / 'envs' / 'files')):
        with pytest.raises(RecordError, match='not a directory of'):
            environment_spec(tmp_path, outside)
    spec = {'path': 'envs/Solver One', 'sha256': 'abcdef0123456789'}
    assert image_name('econ-ark/thing', spec, whole=False) == 'econ-ark/thing-envs-solver-one:abcdef012345'
    assert image_name('econ-ark/thing', spec) == 'econ-ark/thing:abcdef012345'


def test_a_lock_repo2docker_does_not_install_from_is_refused(tmp_path):
    locked = tmp_path / 'envs' / 'locked'
    locked.mkdir(parents=True)
    (locked / 'requirements.txt').write_text('cowsay\n')
    assert environment_spec(tmp_path, 'envs/locked') == ('envs/locked', 'repo2docker', False)
    # repo2docker would install cowsay at whatever version the day gives, and never open the lock
    (locked / 'uv.lock').write_text('version = 1\n')
    with pytest.raises(RecordError, match=r'envs/locked holds uv\.lock, which repo2docker does not install from'):
        environment_spec(tmp_path, 'envs/locked')
    (locked / 'poetry.lock').write_text('')
    with pytest.raises(RecordError, match=r'holds uv\.lock and poetry\.lock'):
        environment_spec(tmp_path, 'envs/locked')
    # a Dockerfile is built as it is written: what it does with the lock is its author's
    (locked / 'binder').mkdir()
    (locked / 'binder' / 'Dockerfile').write_text('FROM scratch\n')
    assert environment_spec(tmp_path, 'envs/locked') == ('envs/locked', 'repo2docker', False)
    (locked / 'Dockerfile').write_text('FROM scratch\n')
    assert environment_spec(tmp_path, 'envs/locked') == ('envs/locked', 'dockerfile', False)
    # the repository's own environment is the whole checkout's, whose root may well hold a lock
    (tmp_path / 'uv.lock').write_text('version = 1\n')
    assert environment_spec(tmp_path) == ('.', 'repo2docker', True)


def test_nothing_is_built_or_run_where_the_facts_cannot_be_read(tmp_path, checkout):
    for outcome in (build(tmp_path, 'conda'), execute(tmp_path, 'conda', 'reproduce.sh')):  # not a git repository
        assert outcome.returncode == 1 and outcome.record is None and 'not a git repository' in outcome.problem
    (checkout / 'binder' / 'Dockerfile').unlink()
    commit_all(checkout, 'no environment')
    assert 'records no file under binder' in build(checkout, 'conda').problem


def test_limits_are_refused_outside_docker(checkout):
    for limit in ({'cpus': 2}, {'memory': '8g'}, {'threads': 2}, {'timeout': 60}, {'network': False},
                  {'read_only': True}, {'mounts': {'/out': checkout}}, {'environment': 'binder'}):
        outcome = execute(checkout, 'conda', 'reproduce.sh', **limit)
        assert outcome.returncode == 1 and 'docker only' in outcome.problem
    assert 'docker only' in build(checkout, 'conda', environment='binder').problem
    with pytest.raises(TypeError, match='readonly'):
        execute(checkout, 'docker', 'reproduce.sh', readonly=True)  # a misspelled option is not dropped


CONDA_STUB = """#!/bin/sh
# Stands in for conda: `--version`, `env update -f <file> --prefix <dir>`, `run -p <dir> <command...>`.
case "$1" in
  --version) echo "conda 99.0.0" ;;
  env) [ -z "$CONDA_STUB_FAILS" ] || { echo "the solve failed"; exit 3; }
       mkdir -p .condaenv; echo "updated from $4" ;;
  run) shift 3; exec "$@" ;;
esac
"""


def test_conda_steps_record_a_host_environment(checkout, tmp_path, monkeypatch):
    stub = tmp_path / 'bin'
    stub.mkdir()
    (stub / 'conda').write_text(CONDA_STUB)
    (stub / 'conda').chmod(0o755)
    monkeypatch.setenv('PATH', f'{stub}:{os.environ["PATH"]}')
    monkeypatch.setenv('SHELL', '/bin/sh')
    (checkout / 'binder' / 'environment.yml').write_text('dependencies: []\n')
    (checkout / 'reproduce.sh').write_text('echo "ran $1"\n')
    commit = commit_all(checkout, 'a conda environment')
    folder = tmp_path / 'records'

    built = build(checkout, 'conda', records_dir=folder)
    assert built.returncode == 0 and built.image is None and built.problem is None, built.stdout
    record = json.loads(built.record_path.read_text())
    records.validate(record)
    assert record['kind'] == 'build' and 'image' not in record and record['where']['runtime'] == 'none'
    assert record['environment']['kind'] == 'host-conda' and record['environment']['builder'] == 'conda 99.0.0'
    assert record['repository']['commit'] == commit

    ran = execute(checkout, 'conda', 'reproduce.sh', arguments=['quickly'], records_dir=folder)
    assert ran.returncode == 0 and ran.stdout == 'ran quickly\n'
    records.validate(ran.record)
    assert ran.record['arguments'] == ['quickly'] and 'image' not in ran.record
    assert ran.record['conditions'] == {'network': True}
    assert ran.record['repository']['dirty'] is False  # .condaenv ignores itself

    # a host build that fails has no image to lack: it is recorded, with its exit code
    monkeypatch.setenv('CONDA_STUB_FAILS', '1')
    failed = build(checkout, 'conda', records_dir=folder)
    assert failed.returncode == 3 and failed.record['outcome'] == {
        'exit_code': 3, 'termination': 'exited', 'log': failed.record_path.with_suffix('.log').name}
    assert 'the solve failed' in (folder / failed.record['outcome']['log']).read_text()


@pytest.mark.docker
@needs_repo2docker
def test_build_then_execute_leave_records(checkout, name, tmp_path):
    folder = tmp_path / 'records'
    commit = git('rev-parse', 'HEAD', cwd=checkout)

    built = build(checkout, 'docker', name=name, records_dir=folder)
    assert built.returncode == 0 and built.problem is None, built.stdout
    assert IMAGE_ID.match(built.image)
    record = json.loads(built.record_path.read_text())
    assert record == built.record
    records.validate(record)
    assert record['kind'] == 'build' and record['repository'] == {'remote': REMOTE, 'commit': commit, 'dirty': False}
    assert record['environment']['path'] == 'binder' and record['environment']['kind'] == 'repo2docker'
    assert record['environment']['builder'].startswith('repo2docker ')
    assert record['image'] == {'name': f'{name}:{record["environment"]["sha256"][:12]}', 'id': built.image}
    assert record['where']['runtime'].startswith('docker ') and record['outcome']['exit_code'] == 0
    assert (folder / record['outcome']['log']).read_text() == built.stdout
    # the image itself says what it was built from
    assert image_label(built.image, COMMIT_LABEL) == commit
    assert image_label(built.image, ENVIRONMENT_LABEL) == record['environment']['sha256']

    ran = execute(checkout, 'docker', 'reproduce.sh', name=name, records_dir=folder, threads=2, cpus=1,
                  memory='256m', network=False, timeout=120, extensions={'a-project': {'cell': 'one'}})
    assert ran.returncode == 0 and ran.problem is None, ran.stdout
    assert 'ran with OMP=2' in ran.stdout and (checkout / 'made.txt').exists()
    record = json.loads(ran.record_path.read_text())
    records.validate(record)
    assert record['kind'] == 'execution' and record['script'] == 'reproduce.sh'
    assert record['image'] == {'id': built.image, 'build_record': built.record_path.name}
    assert record['repository']['dirty'] is False  # the state it started from
    assert record['conditions'] == {'network': False, 'cpus': 1, 'memory_bytes': 256 * 1024 ** 2, 'wall_seconds': 120,
                                    'thread_variables': {v: '2' for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                                          'MKL_NUM_THREADS')}}
    assert record['outcome']['termination'] == 'exited' and record['when']['duration_seconds'] > 0
    assert record['extensions'] == {'a-project': {'cell': 'one'}}
    assert (folder / record['outcome']['log']).read_text() == ran.stdout

    # a second run starts from a checkout the first one wrote into: it runs, and says so
    again = execute(checkout, 'docker', 'reproduce.sh', name=name, records_dir=folder)
    assert again.returncode == 0, again.stdout
    assert again.record['repository']['dirty'] is True
    names = git('ls-tree', '-r', '--name-only', again.record['repository']['worktree_tree'], cwd=checkout)
    assert 'made.txt' in names.splitlines()
    assert again.record['environment'] == built.record['environment'] and again.record['image']['id'] == built.image
    assert again.record_path != ran.record_path and len(list(folder.glob('*_execution.json'))) == 2


@pytest.mark.docker
@needs_repo2docker
def test_an_image_built_from_something_else_is_refused(checkout, name, tmp_path):
    folder = tmp_path / 'records'
    assert 'build it first' in execute(checkout, 'docker', 'reproduce.sh', name=name).problem  # nothing built yet

    first = build(checkout, 'docker', name=name, records_dir=folder)
    assert first.returncode == 0, first.stdout

    # the environment changes: no image has its name
    (checkout / 'binder' / 'Dockerfile').write_text('FROM bash:5.2\nENV CHANGED=1\n')
    refused = execute(checkout, 'docker', 'reproduce.sh', name=name, records_dir=folder)
    assert refused.returncode == 1 and 'build it first' in refused.problem and not (checkout / 'made.txt').exists()
    git('checkout', '--quiet', '--', 'binder/Dockerfile', cwd=checkout)

    # the commit changes and the environment does not: the name is the same, the image is of the old commit
    (checkout / 'reproduce.sh').write_text('echo "the later script"\n')
    later = commit_all(checkout, 'later')
    refused = execute(checkout, 'docker', 'reproduce.sh', name=name, records_dir=folder)
    assert refused.returncode == 1 and 'build it again' in refused.problem and refused.record is None

    # build always builds, where repo2docker alone would keep the image the name already has
    second = build(checkout, 'docker', name=name, records_dir=folder)
    assert second.returncode == 0 and second.image != first.image, second.stdout
    assert second.record['image']['name'] == first.record['image']['name']
    assert image_label(second.image, COMMIT_LABEL) == later
    ran = execute(checkout, 'docker', 'reproduce.sh', name=name, records_dir=folder)
    assert ran.returncode == 0 and 'the later script' in ran.stdout
    assert ran.record['image'] == {'id': second.image, 'build_record': second.record_path.name}

    # the first image lost its name to the second and is still there, until clean removes both
    assert image_id(first.image) == first.image
    clean_docker(name)
    assert image_id(first.image) is None and image_id(second.image) is None


@pytest.mark.docker
@needs_repo2docker
def test_environments_in_directories_of_their_own(checkout, name, tmp_path):
    folder, job = tmp_path / 'records', tmp_path / 'job'
    job.mkdir()
    (checkout / 'envs' / 'solver').mkdir(parents=True)
    (checkout / 'envs' / 'solver' / 'Dockerfile').write_text('FROM bash:5.2\nENV SOLVER=one\n')
    (checkout / 'envs' / 'files' / 'binder').mkdir(parents=True)
    (checkout / 'envs' / 'files' / 'binder' / 'Dockerfile').write_text('FROM bash:5.2\nENV FILES=two\n')
    (checkout / 'job.sh').write_text('echo "solver=${SOLVER:-} files=${FILES:-} level=$1" | tee /out/result.txt\n')
    commit_all(checkout, 'two environments and a job')

    # a Dockerfile directory: built alone, by docker
    solver = build(checkout, 'docker', environment='envs/solver', name=name, records_dir=folder)
    assert solver.returncode == 0 and solver.problem is None, solver.stdout
    record = solver.record
    assert record['environment']['path'] == 'envs/solver' and record['environment']['kind'] == 'dockerfile'
    assert record['environment']['builder'] == record['where']['runtime']
    assert record['image']['name'] == f'{name}-envs-solver:{record["environment"]["sha256"][:12]}'
    assert image_label(solver.image, COMMIT_LABEL) is None  # it holds nothing of the checkout

    # a directory of binder files: built alone, by repo2docker
    files = build(checkout, 'docker', environment='envs/files', name=name, records_dir=folder)
    assert files.returncode == 0 and files.problem is None, files.stdout
    assert files.record['environment']['kind'] == 'repo2docker' and files.image != solver.image
    assert files.record['image']['name'].startswith(f'{name}-envs-files:')

    # the checkout moves on, the environments do not: no rebuild is asked for
    (checkout / 'README.md').write_text('later\n')
    commit_all(checkout, 'later')
    contract = dict(name=name, records_dir=folder, target='/src', read_only=True, mounts={'/out': job},
                    tmpfs=['/tmp'], variables={'HOME': '/tmp'}, user='host', network=False, jit_cache='cold')
    ran = execute(checkout, 'docker', 'job.sh', arguments=['low'], environment='envs/solver', **contract)
    assert ran.returncode == 0 and ran.stdout == 'solver=one files= level=low\n', ran.stdout
    assert (job / 'result.txt').read_text() == ran.stdout
    records.validate(ran.record)
    assert ran.record['image'] == {'id': solver.image, 'build_record': solver.record_path.name}
    assert ran.record['environment'] == solver.record['environment']
    assert ran.record['conditions'] == {'network': False, 'jit_cache': 'cold'}
    assert ran.record['repository']['dirty'] is False  # the checkout was read-only: nothing was written into it
    other = execute(checkout, 'docker', 'job.sh', arguments=['low'], environment='envs/files', **contract)
    assert other.returncode == 0 and other.stdout == 'solver= files=two level=low\n', other.stdout

    # the environment changes: its image has another name, which nothing has yet
    (checkout / 'envs' / 'solver' / 'Dockerfile').write_text('FROM bash:5.2\nENV SOLVER=three\n')
    refused = execute(checkout, 'docker', 'job.sh', environment='envs/solver', **contract)
    assert refused.returncode == 1 and 'build it first' in refused.problem

    # clean finds every environment's images under the one name
    clean_docker(name)
    assert image_id(solver.image) is None and image_id(files.image) is None


@pytest.mark.docker
@needs_repo2docker
def test_a_directory_repo2docker_finds_no_environment_in_is_not_built(checkout, name, tmp_path):
    # Left to its default, repo2docker builds a bare Python image from such a directory and
    # exits 0; the record would then carry the hash of files that nothing installed from.
    folder, unread, locked = tmp_path / 'records', checkout / 'envs' / 'unread', checkout / 'envs' / 'locked'
    unread.mkdir(parents=True)
    (unread / 'pyproject.toml').write_text('[project]\nname = "unread"\nversion = "0"\ndependencies = ["cowsay==6.1"]\n')
    (unread / '.python-version').write_text('3.12.3\n')
    locked.mkdir()
    (locked / 'requirements.txt').write_text('cowsay\n')
    (locked / 'uv.lock').write_text('version = 1\n')
    commit_all(checkout, 'two environments repo2docker cannot build as they are meant')
    assert environment_spec(checkout, 'envs/unread') == ('envs/unread', 'repo2docker', False)

    def images(directory):
        listed = subprocess.run(['docker', 'images', '--quiet', '--filter', f'reference={name}-envs-{directory}'],
                                capture_output=True, text=True)
        return listed.stdout.split()

    # nothing repo2docker builds from: repo2docker itself refuses, and there is no image to record
    built = build(checkout, 'docker', environment='envs/unread', name=name, records_dir=folder)
    assert built.returncode != 0 and built.image is None and built.record is None, built.stdout
    assert 'found nothing to build an environment from in envs/unread' in built.problem
    assert 'Dockerfile' in built.problem and images('unread') == []

    # a lock repo2docker would not install from: nothing is built, or run
    built = build(checkout, 'docker', environment='envs/locked', name=name, records_dir=folder)
    assert built.returncode != 0 and built.image is None and built.record is None
    assert built.problem.startswith('not built: envs/locked holds uv.lock') and images('locked') == []
    assert 'repo2docker' not in built.stdout.replace(built.problem, '')  # it was never started
    ran = execute(checkout, 'docker', 'reproduce.sh', environment='envs/locked', name=name, records_dir=folder)
    assert ran.returncode != 0 and ran.record is None and 'envs/locked holds uv.lock' in ran.problem
    assert not folder.exists() or list(folder.glob('*.json')) == []


@pytest.mark.docker
@needs_repo2docker
def test_a_run_is_reported_when_its_record_cannot_be_written(checkout, name, tmp_path):
    built = build(checkout, 'docker', name=name, remote='file:///somewhere/thing', records_dir=tmp_path / 'records')
    assert built.returncode == 0 and IMAGE_ID.match(built.image)
    assert built.record is None and built.record_path is None and 'no record' in built.problem
    assert list((tmp_path / 'records').glob('*.json')) == []


@pytest.mark.docker
@needs_repo2docker
def test_the_command_line_writes_records(tmp_path, monkeypatch, capsys):
    # A catalog entry at an https remote that git is told to read from a local repository.
    origin = make_repository(tmp_path / 'origin')
    git('tag', '-a', 'v1.0.0', '-m', 'v1.0.0', cwd=origin)
    entry = f'RecordTest{uuid4().hex[:10]}'
    remote = f'https://example.org/test/{entry}'
    config = tmp_path / 'gitconfig'
    config.write_text(f'[url "{origin.as_uri()}"]\n\tinsteadOf = {remote}\n')
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(config))
    root = tmp_path / 'REMARK'
    (root / 'REMARKs').mkdir(parents=True)
    (root / 'REMARKs' / f'{entry}.yml').write_text(f'name: {entry}\nremote: {remote}\ntitle: {entry}\ntag: v1.0.0\n')
    monkeypatch.chdir(root)
    try:
        main(['pull', f'REMARKs/{entry}.yml'])
        with pytest.raises(SystemExit) as e:  # nothing built yet: execute says so and exits non-zero
            main(['execute', 'docker', f'REMARKs/{entry}.yml'])
        assert f'execute failed for: {entry}' in str(e.value.code)
        assert 'build it first' in capsys.readouterr().out

        main(['build', '--no-host', 'docker', f'REMARKs/{entry}.yml'])
        main(['execute', '--threads', '2', 'docker', f'REMARKs/{entry}.yml'])
        folder = root / '_REMARK' / 'records' / entry
        (built,), (ran,) = sorted(folder.glob('*_build.json')), sorted(folder.glob('*_execution.json'))
        build_record, execution = json.loads(built.read_text()), json.loads(ran.read_text())
        for record in (build_record, execution):
            records.validate(record)
            assert record['repository']['remote'] == remote and record['repository']['tag'] == 'v1.0.0'
        assert 'host' not in build_record['where'] and 'host' in execution['where']
        assert build_record['image']['name'].startswith(f'econ-ark/{entry.lower()}:')
        assert execution['image'] == {'id': build_record['image']['id'], 'build_record': built.name}
        assert execution['script'] == 'reproduce.sh' and execution['outcome']['exit_code'] == 0
        assert str(ran) in capsys.readouterr().out  # the command says where the record is
        # the latest log and return code are still where `logs` reads them
        assert (root / '_REMARK' / 'logs' / 'execute' / f'{entry}_docker_rc.log').read_text() == '0'
    finally:
        main(['clean', 'docker', f'REMARKs/{entry}.yml'])
    listed = subprocess.run(['docker', 'images', '--quiet', f'econ-ark/{entry.lower()}'], capture_output=True, text=True)
    assert listed.stdout.strip() == ''

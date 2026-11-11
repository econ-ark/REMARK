"""The facts a record holds, read from git and the machine, and the schema's refusals."""

import hashlib
import json
import os
import re

import pytest
from jsonschema import Draft202012Validator

from conftest import git
from remark import records
from remark.records import RecordError

REMOTE = 'https://example.org/test/thing'
STAMP = re.compile(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00$')


@pytest.fixture
def repo(tmp_path):
    """A committed repository with an environment in binder/: a file, an executable, a link; and an ignore rule."""
    repo = tmp_path / 'thing'
    (repo / 'binder').mkdir(parents=True)
    git('init', '--quiet', '--initial-branch', 'main', cwd=repo)
    git('config', 'user.email', 'test@example.org', cwd=repo)
    git('config', 'user.name', 'test', cwd=repo)
    git('remote', 'add', 'origin', 'git@example.org:test/thing.git', cwd=repo)
    (repo / 'binder' / 'environment.yml').write_text('dependencies:\n  - python=3.12\n')
    (repo / 'binder' / 'postBuild').write_text('#!/bin/bash\necho built\n')
    (repo / 'binder' / 'postBuild').chmod(0o755)
    (repo / 'binder' / 'requirements.txt').symlink_to('environment.yml')
    (repo / 'reproduce.sh').write_text('echo ran\n')
    (repo / '.gitignore').write_text('scratch/\n*.cache\n')
    git('add', '-A', cwd=repo)
    git('commit', '--quiet', '-m', 'first', cwd=repo)
    return repo


def sha(data):
    return hashlib.sha256(data).hexdigest()


def listing_hash(files):
    """The schema's hash, computed apart from the package: {path: (type, mode, bytes)}."""
    lines = ''.join(f'{kind} {mode} {sha(data)} {path}\n'
                    for path, (kind, mode, data) in sorted(files.items(), key=lambda item: item[0].encode()))
    return sha(lines.encode())


BINDER = {
    'environment.yml': ('blob', '100644', b'dependencies:\n  - python=3.12\n'),
    'postBuild': ('blob', '100755', b'#!/bin/bash\necho built\n'),
    'requirements.txt': ('link', '120000', b'environment.yml'),
}


def test_remote_is_normalized_to_https():
    for url in ('git@example.org:test/thing.git', 'ssh://git@example.org/test/thing.git',
                'git://example.org/test/thing', 'https://example.org/test/thing.git', 'https://example.org/test/thing/'):
        assert records.https_remote(url) == REMOTE


def test_a_clean_repository(repo):
    block = records.repository(repo)
    assert block == {'remote': REMOTE, 'commit': git('rev-parse', 'HEAD', cwd=repo), 'dirty': False}
    assert records.repository(repo, remote='https://example.org/the/catalog.git')['remote'] == 'https://example.org/the/catalog'


def test_the_tag_is_recorded_only_when_head_is_at_it(repo):
    git('tag', '-a', 'v1.0.0', '-m', 'v1.0.0', cwd=repo)
    assert records.repository(repo, tag='v1.0.0')['tag'] == 'v1.0.0'
    assert 'tag' not in records.repository(repo, tag='v9.9.9')
    (repo / 'later.txt').write_text('later\n')
    git('add', '-A', cwd=repo)
    git('commit', '--quiet', '-m', 'later', cwd=repo)
    assert 'tag' not in records.repository(repo, tag='v1.0.0')


def test_a_dirty_repository_records_its_working_tree_and_leaves_the_index_alone(repo):
    (repo / 'reproduce.sh').write_text('echo changed\n')   # a tracked change, not staged
    (repo / 'made.txt').write_text('output\n')              # untracked
    (repo / 'scratch').mkdir()
    (repo / 'scratch' / 'big').write_text('ignored\n')      # ignored
    before = git('status', '--porcelain', cwd=repo)
    block = records.repository(repo)
    assert block['dirty'] is True
    names = git('ls-tree', '-r', '--name-only', block['worktree_tree'], cwd=repo).splitlines()
    assert 'made.txt' in names and 'scratch/big' not in names
    assert git('cat-file', 'blob', f'{block["worktree_tree"]}:reproduce.sh', cwd=repo) == 'echo changed'
    # the real index was neither read nor changed: nothing is staged, and status is as it was
    assert git('diff', '--cached', '--name-only', cwd=repo) == ''
    assert git('status', '--porcelain', cwd=repo) == before
    assert records.tree_that_ran(block) == block['worktree_tree']


def test_staged_changes_do_not_stand_in_for_the_working_tree(repo):
    (repo / 'reproduce.sh').write_text('echo staged\n')
    git('add', 'reproduce.sh', cwd=repo)
    (repo / 'reproduce.sh').write_text('echo on disk\n')
    tree = records.repository(repo)['worktree_tree']
    assert git('cat-file', 'blob', f'{tree}:reproduce.sh', cwd=repo) == 'echo on disk'
    assert git('diff', '--cached', '--name-only', cwd=repo) == 'reproduce.sh'


def test_environment_hash_is_the_listing_of_what_git_records(repo):
    block = records.repository(repo)
    env = records.environment(repo, block, 'binder/', 'repo2docker', 'repo2docker 2026.4.0')
    assert env == {'path': 'binder', 'kind': 'repo2docker', 'sha256': listing_hash(BINDER),
                   'git_tree': git('rev-parse', 'HEAD:binder', cwd=repo), 'builder': 'repo2docker 2026.4.0'}


def test_environment_hash_ignores_what_git_ignores_and_sees_what_it_would_add(repo):
    clean = records.environment(repo, records.repository(repo), 'binder', 'repo2docker')['sha256']
    (repo / 'binder' / 'pip.cache').write_text('ignored\n')
    (repo / 'made.txt').write_text('outside the environment\n')
    block = records.repository(repo)
    assert block['dirty'] and records.environment(repo, block, 'binder', 'repo2docker')['sha256'] == clean
    (repo / 'binder' / 'apt.txt').write_text('git\n')  # untracked, not ignored: a build would read it
    changed = records.environment(repo, records.repository(repo), 'binder', 'repo2docker')['sha256']
    assert changed == listing_hash({**BINDER, 'apt.txt': ('blob', '100644', b'git\n')}) != clean


def test_environment_at_the_root(repo):
    everything = {f'binder/{name}': entry for name, entry in BINDER.items()}
    everything['reproduce.sh'] = ('blob', '100644', b'echo ran\n')
    everything['.gitignore'] = ('blob', '100644', b'scratch/\n*.cache\n')
    env = records.environment(repo, records.repository(repo), '.', 'repo2docker')
    assert env['path'] == '.' and env['sha256'] == listing_hash(everything)
    assert env['git_tree'] == git('rev-parse', 'HEAD^{tree}', cwd=repo)


def test_a_directory_git_records_nothing_under_is_refused(repo, tmp_path):
    with pytest.raises(RecordError, match='records no file under env'):
        records.environment(repo, records.repository(repo), 'env', 'repo2docker')
    with pytest.raises(RecordError):
        records.repository(tmp_path)  # not a git repository


def test_exit_codes_that_name_a_signal():
    assert records.by_signal(0) == records.by_signal(1) == records.by_signal(124) == ('exited', None)
    assert records.by_signal(255) == ('exited', None)
    assert records.by_signal(137) == ('signal', 'SIGKILL')
    assert records.by_signal(-15) == ('signal', 'SIGTERM')


def test_timed_run(tmp_path):
    done = records.timed_run(['sh', '-c', 'echo out; echo err >&2; exit 3'])
    assert (done.returncode, done.termination, done.signal) == (3, 'exited', None)
    assert 'out' in done.stdout and 'err' in done.stdout
    assert STAMP.match(done.started) and STAMP.match(done.ended) and done.duration_seconds >= 0
    killed = records.timed_run(['sh', '-c', 'kill -TERM $$'])
    assert (killed.returncode, killed.termination, killed.signal) == (-15, 'signal', 'SIGTERM')
    called = []
    late = records.timed_run(['sh', '-c', 'echo started; sleep 30'], timeout=1, on_timeout=lambda: called.append(1))
    assert (late.returncode, late.termination, called) == (124, 'timeout', [1])
    assert 'started' in late.stdout and 'TIMEOUT' in late.stdout and late.duration_seconds < 20


def test_memory_limits_as_bytes():
    assert records.memory_bytes('8g') == 8 * 1024 ** 3
    assert records.memory_bytes('512M') == 512 * 1024 ** 2
    assert records.memory_bytes(4096) == records.memory_bytes('4096b') == 4096
    assert records.memory_bytes('1.5gb') == records.memory_bytes('1.5GiB') == 3 * 1024 ** 3 // 2
    with pytest.raises(RecordError):
        records.memory_bytes('lots')


def test_the_machine(monkeypatch):
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    where = records.machine('none')
    assert {'cpu', 'cores', 'memory_bytes', 'os', 'kernel', 'runtime'} <= set(where)
    assert where['cores'] >= 1 and where['memory_bytes'] > 0 and where['cpu'] and 'ci' not in where
    assert 'host' not in records.machine('none', host=False)
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('GITHUB_RUN_ID', '42')
    monkeypatch.setenv('GITHUB_SERVER_URL', 'https://github.com')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'econ-ark/REMARK')
    assert records.machine('none')['ci'] == {'provider': 'github-actions', 'run_id': '42',
                                             'url': 'https://github.com/econ-ark/REMARK/actions/runs/42'}


# The schema -----------------------------------------------------------------------------

IMAGE = 'sha256:' + 'a' * 64


def blocks(repo, runtime='docker 29.1.3'):
    block = records.repository(repo)
    kind = 'host-conda' if runtime == 'none' else 'repo2docker'
    return dict(repository=block, environment=records.environment(repo, block, 'binder', kind),
                where=records.machine(runtime), result=records.timed_run(['true']))


def test_the_schema_is_a_schema_of_its_own():
    schema = records.schema()
    Draft202012Validator.check_schema(schema)
    assert schema['$id'] == records.schema_id() == f'https://github.com/econ-ark/REMARK/execution-record/{records.RECORD_VERSION}.json'


def test_records_that_match(repo):
    records.validate(records.build_record(**blocks(repo), image={'name': 'econ-ark/thing:abc', 'id': IMAGE},
                                          conditions={'network': True}, log='build.log'))
    records.validate(records.build_record(**blocks(repo, 'none')))  # a host environment has no image
    execution = records.execution_record(
        **blocks(repo), script='reproduce.sh', arguments=['--quick'], image={'id': IMAGE, 'build_record': 'b.json'},
        conditions={'network': False, 'cpus': 2, 'memory_bytes': 8 * 1024 ** 3, 'wall_seconds': 3600,
                    'thread_variables': {'OMP_NUM_THREADS': '2'}},
        extensions={'a-project': {'anything': [1, 2]}})
    records.validate(execution)
    assert execution['recorder']['name'] == 'econ-ark-remark' and execution['kind'] == 'execution'
    assert execution['$schema'] == records.schema_id() and execution['record_version'] == records.RECORD_VERSION


def test_a_record_of_an_earlier_draft_is_still_read(repo):
    # written by a project's runner under its own schema's name, before this schema had its number
    record = records.execution_record(**blocks(repo, 'none'), script='reproduce.sh', extensions={'a-project': {}})
    record.update({'$schema': 'https://example.org/a-project/runs.registry/0.1.0-draft.6.json',
                   'record_version': '0.1.0-draft.4'})
    records.validate(record)


def test_records_that_are_refused(repo, tmp_path):
    def refused(record, what):
        with pytest.raises(RecordError, match=what):
            records.validate(record)

    refused(records.build_record(**blocks(repo)), 'image')  # built in a container: the image is required
    refused(records.execution_record(**blocks(repo), script='reproduce.sh'), 'image')
    parts = blocks(repo)
    parts['repository'] = {**parts['repository'], 'dirty': True}  # dirty without the working tree's tree
    refused(records.build_record(**parts, image={'name': 'n', 'id': IMAGE}), 'worktree_tree')
    parts = blocks(repo)
    parts['repository'] = {**parts['repository'], 'remote': 'file:///somewhere/thing'}
    refused(records.build_record(**parts, image={'name': 'n', 'id': IMAGE}), 'remote')
    record = records.build_record(**blocks(repo), image={'name': 'n', 'id': 'thing:latest'})  # a name is not an ID
    refused(record, 'image/id')
    record = records.execution_record(**blocks(repo, 'none'), script='reproduce.sh')
    record['duration'] = 3  # nothing outside the schema
    refused(record, 'duration')
    with pytest.raises(RecordError):
        records.write(record, tmp_path / 'record.json')
    assert not (tmp_path / 'record.json').exists()  # an invalid record is never written


def test_write(repo, tmp_path):
    record = records.execution_record(**blocks(repo, 'none'), script='reproduce.sh', log='run.log')
    records.write(record, tmp_path / 'record.json')
    assert json.loads((tmp_path / 'record.json').read_text()) == record
    assert record['outcome'] == {'exit_code': 0, 'termination': 'exited', 'log': 'run.log'}
    assert os.path.basename(str(records.SCHEMA_FILE)) == 'execution-record.schema.json'

"""Execution records: what was built or run, from which commit and environment, on which
machine, when, and how it ended, in the format of schemas/execution-record.schema.json.

Everything here is a fact read from git, the machine or the process; nothing is inferred.
The repository's state is its remote, its commit, whether `git status --porcelain` was
empty and, when it was not, the tree of the working tree written through a temporary
index. An environment's hash is taken over the files git records under its directory in
the tree that ran, so that untracked files and caches never enter it and two
specifications compare file by file.

    from remark.records import repository, environment, machine, write
"""

import hashlib
import json
import os
import platform
import re
import signal as signals
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from subprocess import run, Popen, PIPE, STDOUT, DEVNULL, TimeoutExpired
from time import perf_counter

from remark import __version__

RECORD_VERSION = '0.1.0'
SCHEMA_FILE = Path(__file__).resolve().parent / 'schemas' / 'execution-record.schema.json'


class RecordError(ValueError):
    """A fact a record needs could not be read, or a record does not match the schema."""


def schema():
    return json.loads(SCHEMA_FILE.read_text(encoding='utf-8'))


def schema_id():
    return schema()['$id']


# Reading git ----------------------------------------------------------------------------

def git(*args, cwd, env=None):
    done = run(['git', *args], cwd=cwd, env=env, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    if done.returncode:
        raise RecordError(f'git {" ".join(args)} in {cwd}: {done.stderr.strip()}')
    return done.stdout.strip()


def https_remote(url):
    """A remote in the https form the catalog names: ssh and git forms rewritten, `.git` dropped."""
    u = url.strip()
    if u.startswith('git@') and ':' in u:
        host, _, path = u[4:].partition(':')
        u = f'https://{host}/{path}'
    elif u.startswith('ssh://'):
        u = 'https://' + u[6:].split('@', 1)[-1]
    elif u.startswith('git://'):
        u = 'https://' + u[6:]
    u = u.rstrip('/')
    return u[:-4] if u.endswith('.git') else u


def worktree_tree(local_repo):
    """The tree of the working tree as it is, tracked changes and untracked files that are
    not ignored included. It is written through a temporary index that both git commands
    read, so that the real index is neither read nor changed."""
    with tempfile.TemporaryDirectory(prefix='remark-index-') as scratch:
        env = {**os.environ, 'GIT_INDEX_FILE': str(Path(scratch) / 'index')}
        git('add', '-A', cwd=local_repo, env=env)
        return git('write-tree', cwd=local_repo, env=env)


def repository(local_repo, *, remote=None, tag=None):
    """The record's `repository`: remote, commit, the tag when HEAD is at it, whether the
    working tree was dirty, and its tree when it was.

    `remote` is the catalog's when given, else the clone's origin. `tag` is the tag the
    catalog pins; it is recorded only if HEAD is the commit that tag names.
    """
    commit = git('rev-parse', 'HEAD', cwd=local_repo)
    out = {'remote': https_remote(remote or git('remote', 'get-url', 'origin', cwd=local_repo)), 'commit': commit}
    if tag:
        named = run(['git', 'rev-parse', '--verify', '--quiet', f'refs/tags/{tag}^{{commit}}'],
                    cwd=local_repo, stdout=PIPE, stderr=DEVNULL, encoding='utf-8')
        if named.returncode == 0 and named.stdout.strip() == commit:
            out['tag'] = tag
    out['dirty'] = bool(git('status', '--porcelain', cwd=local_repo))
    if out['dirty']:
        out['worktree_tree'] = worktree_tree(local_repo)
    return out


def tree_that_ran(repo):
    """The tree a `repository` block describes: the working tree's when dirty, else the commit's."""
    return repo['worktree_tree'] if repo['dirty'] else repo['commit']


def blob_hashes(local_repo, objects):
    """SHA-256 of each git blob's bytes, read from git in one process and never held whole."""
    out = {}
    if not objects:
        return out
    with tempfile.TemporaryFile() as names:
        names.write(('\n'.join(objects) + '\n').encode('ascii'))
        names.seek(0)
        with Popen(['git', 'cat-file', '--batch'], cwd=local_repo, stdin=names, stdout=PIPE) as proc:
            for obj in objects:
                header = proc.stdout.readline().split()
                if len(header) != 3 or header[1] != b'blob':
                    proc.kill()
                    raise RecordError(f'git cat-file in {local_repo}: {obj} is not a blob git can read')
                left, digest = int(header[2]), hashlib.sha256()
                while left:
                    chunk = proc.stdout.read(min(left, 1 << 20))
                    if not chunk:
                        proc.kill()
                        raise RecordError(f'git cat-file in {local_repo}: {obj} ended early')
                    digest.update(chunk)
                    left -= len(chunk)
                proc.stdout.read(1)  # the newline after each object
                out[obj] = digest.hexdigest()
    return out


def environment_hash(local_repo, tree, path):
    """The schema's hash of the directory `path` in `tree`, and git's own name for that directory.

    `git ls-tree -r <tree> -- <path>` lists what git records there, never the index or the
    files on disk. Each blob's bytes, or a symbolic link's text, are hashed alone; then the
    listing of lines `<type> <mode> <sha256> <path>`, the path relative to the directory
    and the lines sorted bytewise on it, is hashed. `path` is '.' for the repository's root.
    """
    root = path in ('.', '')
    path = '.' if root else path.strip('/')
    listing = git('ls-tree', '-r', '-z', tree, *([] if root else ['--', path]), cwd=local_repo)
    entries = []
    for line in listing.split('\0'):
        if not line:
            continue
        meta, _, name = line.partition('\t')
        mode, kind, obj = meta.split()
        if kind != 'blob':
            continue  # a submodule's commit: git records no content for it here
        entries.append((name if root else name[len(path) + 1:], mode, obj))
    if not entries:
        raise RecordError(f'git records no file under {path} in {tree[:12]} of {local_repo}')
    hashes = blob_hashes(local_repo, [obj for _, _, obj in entries])
    lines = sorted(
        (name.encode('utf-8'), f'{"link" if mode == "120000" else "blob"} {mode} {hashes[obj]} {name}\n')
        for name, mode, obj in entries
    )
    digest = hashlib.sha256(''.join(text for _, text in lines).encode('utf-8')).hexdigest()
    return digest, git('rev-parse', f'{tree}^{{tree}}' if root else f'{tree}:{path}', cwd=local_repo)


def environment(local_repo, repo, path, kind, builder=None):
    """The record's `environment`: the directory, how it is built, its hash in the tree that ran."""
    path = '.' if path in ('.', '') else path.strip('/')
    digest, git_tree = environment_hash(local_repo, tree_that_ran(repo), path)
    out = {'path': path, 'kind': kind, 'sha256': digest, 'git_tree': git_tree}
    if builder:
        out['builder'] = builder
    return out


# Reading the machine --------------------------------------------------------------------

def _read(path):
    try:
        return Path(path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''


def _output(cmd):
    try:
        done = run(cmd, stdout=PIPE, stderr=STDOUT, stdin=DEVNULL, encoding='utf-8', errors='replace')
    except OSError:
        return ''
    return done.stdout.strip() if done.returncode == 0 else ''


def tool_version(name, *args):
    """'<name> <version>' as the tool itself reports it, or None when it does not answer."""
    out = _output([name, *(args or ['--version'])]).splitlines()
    if not out:
        return None
    version = out[-1].strip()
    return version if version.lower().startswith(name.lower()) else f'{name} {version}'


def docker_runtime():
    """'docker <server version>' when the daemon answers, else None."""
    version = _output(['docker', 'version', '--format', '{{.Server.Version}}'])
    return f'docker {version}' if version else None


def continuous_integration():
    """The `ci` block on a GitHub Actions runner, else None."""
    env = os.environ
    if env.get('GITHUB_ACTIONS') != 'true' or not env.get('GITHUB_RUN_ID'):
        return None
    out = {'provider': 'github-actions', 'run_id': env['GITHUB_RUN_ID']}
    if env.get('GITHUB_SERVER_URL', '').startswith('https://') and env.get('GITHUB_REPOSITORY'):
        out['url'] = f'{env["GITHUB_SERVER_URL"]}/{env["GITHUB_REPOSITORY"]}/actions/runs/{env["GITHUB_RUN_ID"]}'
    return out


def machine(runtime, *, host=True):
    """The record's `where`: the machine's facts and the runtime, nothing of the run's settings.

    `runtime` is the container runtime with its version, or 'none' on the host. Pass
    host=False to leave the machine's name out.
    """
    cpu = next((ln.split(':', 1)[1].strip() for ln in _read('/proc/cpuinfo').splitlines()
                if ln.startswith('model name')), '')
    cpu = cpu or _output(['sysctl', '-n', 'machdep.cpu.brand_string']) or platform.processor() or platform.machine()
    try:
        memory = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (ValueError, OSError, AttributeError):
        memory = 0
    system = next((ln.split('=', 1)[1].strip().strip('"') for ln in _read('/etc/os-release').splitlines()
                   if ln.startswith('PRETTY_NAME=')), platform.platform())
    out = {'host': platform.node()} if host and platform.node() else {}
    out.update(cpu=cpu, cores=os.cpu_count() or 1, memory_bytes=memory, os=system,
               kernel=platform.release(), runtime=runtime)
    ci = continuous_integration()
    if ci:
        out['ci'] = ci
    return out


def memory_bytes(spec):
    """Docker's memory limit, such as 8g, 512m or 1.5gb, as bytes."""
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?', str(spec).strip().lower())
    if not m:
        raise RecordError(f'memory limit {spec!r}: give a number with b, k, m, g or t')
    return int(float(m.group(1)) * 1024 ** ' kmgt'.index(m.group(2) or ' '))


# Running a process ----------------------------------------------------------------------

def now():
    """An RFC 3339 stamp with an offset, to the second."""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def by_signal(code):
    """('signal', its name) when an exit code says a signal ended the process, else ('exited', None).

    A process the caller started reports a signal as a negative code; a shell, `conda run`
    and docker report it as 128 plus the signal's number.
    """
    number = -code if code < 0 else code - 128
    try:
        return ('signal', signals.Signals(number).name) if number > 0 else ('exited', None)
    except ValueError:
        return 'exited', None


@dataclass
class Run:
    """One process run to its end: its code, its combined output, how it ended and when."""
    args: list
    returncode: int
    stdout: str
    termination: str = 'exited'  # exited, timeout, oom or signal
    signal: str = None
    started: str = None
    ended: str = None
    duration_seconds: float = None

    def when(self):
        return {'started': self.started, 'ended': self.ended, 'duration_seconds': self.duration_seconds}


def timed_run(cmd, *, timeout=None, on_timeout=None, **kwargs):
    """Run `cmd` without a terminal, its output and errors combined, timed on a monotonic clock.

    A run stopped at `timeout` seconds returns 124 with termination 'timeout';
    `on_timeout` is then called, for what stopping the process leaves behind.
    """
    cmd = [str(c) for c in cmd]
    started, clock = now(), perf_counter()
    try:
        done = run(cmd, stdout=PIPE, stderr=STDOUT, stdin=DEVNULL, encoding='utf-8', errors='replace',
                   timeout=timeout, **kwargs)
        code, output, termination, name = done.returncode, done.stdout, *by_signal(done.returncode)
    except TimeoutExpired as e:
        if on_timeout:
            on_timeout()
        output = e.stdout or ''
        if isinstance(output, bytes):
            output = output.decode('utf-8', errors='replace')
        output += f'\nTIMEOUT: stopped after {timeout} s\n'
        code, termination, name = 124, 'timeout', None
    duration = round(perf_counter() - clock, 3)
    return Run(cmd, code, output, termination, name, started, now(), duration)


# The records ----------------------------------------------------------------------------

def recorder():
    """The record's `recorder`: this package, its version and, when it runs from a clean
    checkout of its repository, that checkout's commit."""
    out = {'name': 'econ-ark-remark', 'version': __version__}
    checkout = Path(__file__).resolve().parent.parent
    if checkout.joinpath('.git').exists() and checkout.joinpath('pyproject.toml').exists():
        try:
            if not git('status', '--porcelain', cwd=checkout):  # a changed checkout is not its commit's code
                out['commit'] = git('rev-parse', 'HEAD', cwd=checkout)
        except (RecordError, OSError):
            pass
    return out


def _outcome(result, log):
    out = {'exit_code': result.returncode, 'termination': result.termination}
    if result.signal:
        out['signal'] = result.signal
    if log:
        out['log'] = log
    return out


def build_record(*, repository, environment, where, result, image=None, conditions=None, log=None):
    """A build's record. `result` is the build's Run; `image` is {'name', 'id'} for an image."""
    record = {'$schema': schema_id(), 'record_version': RECORD_VERSION, 'kind': 'build', 'recorder': recorder(),
              'repository': repository, 'environment': environment}
    if image:
        record['image'] = image
    record['where'] = where
    if conditions:
        record['conditions'] = conditions
    record.update(when=result.when(), outcome=_outcome(result, log))
    return record


def execution_record(*, repository, script, environment, where, result, arguments=None, image=None,
                     conditions=None, log=None, extensions=None):
    """An execution's record. `image` is {'id'} with, when known, the 'build_record' path."""
    record = {'$schema': schema_id(), 'record_version': RECORD_VERSION, 'kind': 'execution', 'recorder': recorder(),
              'repository': repository, 'script': script}
    if arguments:
        record['arguments'] = [str(a) for a in arguments]
    record['environment'] = environment
    if image:
        record['image'] = image
    record['where'] = where
    if conditions:
        record['conditions'] = conditions
    record.update(when=result.when(), outcome=_outcome(result, log))
    if extensions:
        record['extensions'] = extensions
    return record


def validate(record):
    """Raise RecordError unless the record matches the schema."""
    from jsonschema import Draft202012Validator

    errors = sorted(Draft202012Validator(schema()).iter_errors(record), key=lambda e: list(e.absolute_path))
    if errors:
        raise RecordError('the record does not match execution-record.schema.json: ' + '; '.join(
            f'{"/".join(map(str, e.absolute_path)) or "record"}: {e.message}' for e in errors[:5]))


def write(record, path):
    """Validate the record, then write it; an invalid record is never written."""
    validate(record)
    Path(path).write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')

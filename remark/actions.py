"""Build and execute with a record: each function does the step, writes what it did in the
format of schemas/execution-record.schema.json, and returns both.

    from remark.actions import build, execute

    built = build(checkout, 'docker', records_dir=folder)          # built.image is the image ID
    ran = execute(checkout, 'docker', 'reproduce.sh', records_dir=folder)
    ran.returncode, ran.record['outcome'], ran.record_path

A docker image is named for the hash of the environment's files, econ-ark/<name>:<hash>,
so a checkout whose environment changed finds no image and must be built again. The
image also carries the commit it was built at, since repo2docker copies the whole
checkout into it: execute refuses an image built at another commit.

A repository can keep further environments in directories of their own, each built with
that directory alone as its context: `environment='envs/solver'` builds its Dockerfile,
or its binder files when it has no Dockerfile, into econ-ark/<name>-envs-solver:<hash>.
Such an image holds nothing else of the checkout, so its name says all it depends on.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from remark import conda, docker, records
from remark.records import RecordError

ENVIRONMENT_DIRECTORIES = ('binder', '.binder')
# Lock files repo2docker installs from none of (Pipfile.lock is the one it reads, as of
# 2026.4.0). A directory it built that held one would be recorded under the lock's hash
# while its image was of whatever versions the day's index gave.
UNREAD_LOCKS = ('uv.lock', 'poetry.lock', 'pdm.lock', 'pixi.lock')


@dataclass
class Outcome:
    """What a build or an execution did."""
    returncode: int
    stdout: str
    record: dict = None       # the record, when one could be made
    record_path: Path = None  # where it was written, when a records folder was given
    image: str = None         # docker: the image's ID
    problem: str = None       # why nothing ran, or why there is no record


def environment_path(local_repo):
    """The directory repo2docker reads the environment from: binder or .binder when the
    repository has one, else its root."""
    for directory in ENVIRONMENT_DIRECTORIES:
        if Path(local_repo, directory).is_dir():
            return directory
    return '.'


def default_name(local_repo):
    """The default name of a checkout's images, without the tag: econ-ark/<folder name>."""
    return f'econ-ark/{Path(local_repo).name.lower().replace(" ", "_")}'


def environment_spec(local_repo, environment=None):
    """Where an environment is and how it is built: (path, kind, whole).

    With no `environment`, or binder or .binder, it is the repository's own: repo2docker
    on the whole checkout, which it copies into the image (`whole` is True). Any other
    directory is built alone: its Dockerfile when it has one, else its binder files.

    A directory left to repo2docker's own builders is refused when it holds one of
    UNREAD_LOCKS: the lock would be in the environment's hash and in nothing installed.
    A Dockerfile, the directory's or its binder folder's, is built as it is written, so
    what it does with a lock is its author's.
    """
    if environment in (None, '', '.') or str(environment).strip('/') in ENVIRONMENT_DIRECTORIES:
        return environment_path(local_repo), 'repo2docker', True
    path = str(environment).strip('/')
    directory = Path(local_repo, path)
    if Path(path).is_absolute() or '..' in Path(path).parts or not directory.is_dir():
        raise RecordError(f'{environment} is not a directory of {local_repo}')
    if (directory / 'Dockerfile').is_file():
        return path, 'dockerfile', False
    locks = [lock for lock in UNREAD_LOCKS if (directory / lock).is_file()]
    if locks and not any((directory / binder / 'Dockerfile').is_file() for binder in ENVIRONMENT_DIRECTORIES):
        raise RecordError(f'{path} holds {" and ".join(locks)}, which repo2docker does not install from: its image '
                          'would not be of the locked versions. Give the directory a Dockerfile that installs '
                          'from the lock')
    return path, 'repo2docker', False


def image_name(base, environment, whole=True):
    """The image's name for an environment: the base name, the directory when it is not
    the repository's own, and the first 12 digits of the environment's hash as the tag."""
    slug = '' if whole else '-' + re.sub(r'[^a-z0-9_.]+', '-', environment['path'].lower()).strip('-.')
    return f'{base}{slug}:{environment["sha256"][:12]}'


def _refused(problem):
    return Outcome(returncode=1, stdout=problem + '\n', problem=problem)


def _facts(local_repo, runtime, remote, tag, host, environment=None):
    """The repository, environment and machine blocks for a build or a run in `runtime`,
    and whether the environment's image holds the whole checkout."""
    repo = records.repository(local_repo, remote=remote, tag=tag)
    if runtime == 'docker':
        engine = records.docker_runtime()
        if engine is None:
            raise RecordError('the docker daemon does not answer')
        path, kind, whole = environment_spec(local_repo, environment)
        builder = engine if kind == 'dockerfile' else records.tool_version('repo2docker')
        env = records.environment(local_repo, repo, path, kind, builder)
        return repo, env, records.machine(engine, host=host), whole
    if environment is not None:
        raise RecordError('an environment in a directory of its own is built with docker only')
    env = records.environment(local_repo, repo, 'binder', 'host-conda', records.tool_version('conda'))
    return repo, env, records.machine('none', host=host), True


def _save(record, result, records_dir, kind):
    """Write the record and its log into `records_dir`, named for the start time. Returns the record's path."""
    records.validate(record)
    if records_dir is None:
        return None
    records_dir = Path(records_dir)
    records_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.started.replace('-', '').replace(':', '').replace('+0000', 'Z')
    stem, extra = f'{stamp}_{kind}', 1
    while (records_dir / f'{stem}.json').exists():  # two in one second
        extra += 1
        stem = f'{stamp}-{extra}_{kind}'
    (records_dir / f'{stem}.log').write_text(result.stdout, encoding='utf-8')
    record['outcome']['log'] = f'{stem}.log'
    path = records_dir / f'{stem}.json'
    records.write(record, path)
    return path


def build_record_for(records_dir, image):
    """The newest build record in `records_dir` for the image ID, as a path, or None."""
    if records_dir is None or not Path(records_dir).is_dir():
        return None
    for path in sorted(Path(records_dir).glob('*_build.json'), reverse=True):
        try:
            if json.loads(path.read_text(encoding='utf-8')).get('image', {}).get('id') == image:
                return path
        except ValueError:
            continue
    return None


def build(local_repo, runtime, *, environment=None, name=None, remote=None, tag=None, records_dir=None,
          host=True):
    """Build the checkout's environment: a docker image with repo2docker, or ./.condaenv with conda.

    `runtime` is 'docker' or 'conda'. `environment` is a directory of the checkout that
    holds an environment of its own (docker only; see environment_spec); without it the
    repository's own is built. `name` is the images' name without a tag (default
    econ-ark/<folder name>). `remote` and `tag` are the catalog's, for the record; the
    remote defaults to the clone's origin. With `records_dir`, the record and the build's
    log are written there.

    Returns an Outcome. Nothing is built, and `problem` says why, when the facts a record
    needs cannot be read: the checkout is not a git repository, git records no file under
    the environment's directory, or docker does not answer. A docker build that fails has
    no image and so no record; its output is in `stdout`. A directory named as an
    environment in which repo2docker finds nothing to build from fails so, rather than
    yield a bare image recorded under the hash of files that were never read; one that
    holds a lock file repo2docker does not install from is not built at all.
    """
    local_repo = Path(local_repo)
    try:
        repo, env, where, whole = _facts(local_repo, runtime, remote, tag, host, environment)
    except (RecordError, OSError) as e:
        return _refused(f'not built: {e}')
    if runtime == 'docker':
        base = name or default_name(local_repo)
        named = image_name(base, env, whole)
        labels = {docker.NAME_LABEL: base, docker.ENVIRONMENT_LABEL: env['sha256']}
        if whole:  # the image holds the checkout, so it is of this commit
            result = docker.build_docker(local_repo, named, {**labels, docker.COMMIT_LABEL: repo['commit']})
        elif env['kind'] == 'dockerfile':
            result = docker.build_dockerfile(local_repo, named, env['path'], labels)
        else:
            result = docker.build_docker(local_repo, named, labels, subdir=env['path'])
        image = docker.image_id(named) if result.returncode == 0 else None
        if image is None:
            problem = 'the build made no image'
            if not whole and env['kind'] == 'repo2docker' and docker.NOTHING_FOUND in result.stdout:
                problem = (f'repo2docker found nothing to build an environment from in {env["path"]}: it reads '
                           'environment.yml, requirements.txt and its other configuration files, and not '
                           '.python-version or a pyproject.toml without a build system. '
                           'A Dockerfile in the directory is built as it is written')
            return Outcome(result.returncode or 1, result.stdout, problem=problem)
        image_block = {'name': named, 'id': image}
    else:
        result, image, image_block = conda.build_conda(local_repo), None, None
    out = Outcome(result.returncode, result.stdout, image=image)
    try:
        out.record = records.build_record(repository=repo, environment=env, where=where, result=result,
                                          image=image_block, conditions={'network': True})
        out.record_path = _save(out.record, result, records_dir, 'build')
    except (RecordError, OSError) as e:
        out.record, out.problem = None, f'no record: {e}'
    return out


def execute(local_repo, runtime, script, *, arguments=(), environment=None, image=None, name=None, remote=None,
            tag=None, records_dir=None, host=True, network=True, cpus=None, memory=None, threads=None,
            timeout=None, extensions=None, jit_cache=None, **contract):
    """Run `script` from the checkout in its built environment and record the run.

    `runtime` is 'docker' or 'conda'. `environment` is as for `build`. For docker, the image is the one `build` named for
    the environment as it is now, or `image` (a name or an ID) when given; `network`,
    `cpus`, `memory`, `threads` and `timeout` limit the container and are docker's alone.
    `extensions` is a project's own part of the record, one object per namespace.
    `contract` is execute_docker's target, read_only, mounts, tmpfs, variables and user,
    for a runner whose jobs read the checkout and write elsewhere; `jit_cache` ('cold',
    'warm' or 'none') is the caller's statement for the record, since a scratch
    filesystem per run puts a just-in-time compiler's work into every duration.

    Returns an Outcome. Nothing runs, and `problem` says why, when the facts cannot be
    read, when there is no image for the environment as it is, or when the image was
    built by this tool at another commit.
    """
    local_repo = Path(local_repo)
    unknown = set(contract) - {'target', 'read_only', 'mounts', 'tmpfs', 'variables', 'user'}
    if unknown:
        raise TypeError(f'execute() got unexpected keyword arguments: {", ".join(sorted(unknown))}')
    if runtime != 'docker' and (not network or contract or any(
            v is not None for v in (image, cpus, memory, threads, timeout))):
        return _refused('not run: image, the limits and the mounts apply to docker only')
    try:
        repo, env, where, whole = _facts(local_repo, runtime, remote, tag, host, environment)
        conditions = {'network': bool(network)}
        if cpus is not None:
            conditions['cpus'] = cpus
        if memory is not None:
            conditions['memory_bytes'] = records.memory_bytes(memory)
        if threads is not None:
            conditions['thread_variables'] = {variable: str(threads) for variable in docker.THREAD_VARIABLES}
        if timeout is not None:
            conditions['wall_seconds'] = timeout
        if jit_cache is not None:
            conditions['jit_cache'] = jit_cache
    except (RecordError, OSError) as e:
        return _refused(f'not run: {e}')

    image_block = None
    if runtime == 'docker':
        wanted = image or image_name(name or default_name(local_repo), env, whole)
        found = docker.image_id(wanted)
        if found is None:
            return _refused(f'not run: no image {wanted} for {env["path"]} as it is at {repo["commit"][:12]}; '
                            'build it first')
        built_at = docker.image_label(found, docker.COMMIT_LABEL)
        if built_at and built_at != repo['commit']:
            return _refused(f'not run: image {wanted} was built at commit {built_at[:12]} and the checkout is at '
                            f'{repo["commit"][:12]}; build it again')
        image_block = {'id': found}
        build_record = build_record_for(records_dir, found)
        if build_record is not None:
            image_block['build_record'] = build_record.name
        result = docker.execute_docker(local_repo, found, script, arguments=arguments, network=network,
                                       cpus=cpus, memory=memory, threads=threads, timeout=timeout, **contract)
    else:
        found = None
        result = conda.execute_conda(local_repo, script, arguments=arguments)

    out = Outcome(result.returncode, result.stdout, image=found)
    try:
        out.record = records.execution_record(repository=repo, script=script, arguments=arguments,
                                              environment=env, where=where, result=result, image=image_block,
                                              conditions=conditions, extensions=extensions)
        out.record_path = _save(out.record, result, records_dir, 'execution')
    except (RecordError, OSError) as e:
        out.record, out.problem = None, f'no record: {e}'
    return out

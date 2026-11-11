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
"""

import json
from dataclasses import dataclass
from pathlib import Path

from remark import conda, docker, records
from remark.records import RecordError

ENVIRONMENT_DIRECTORIES = ('binder', '.binder')


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


def image_name(repository_name, environment):
    """The image's name for an environment: the first 12 digits of its hash as the tag."""
    return f'{repository_name}:{environment["sha256"][:12]}'


def _refused(problem):
    return Outcome(returncode=1, stdout=problem + '\n', problem=problem)


def _facts(local_repo, runtime, remote, tag, host):
    """The repository, environment and machine blocks for a build or a run in `runtime`."""
    repo = records.repository(local_repo, remote=remote, tag=tag)
    if runtime == 'docker':
        engine = records.docker_runtime()
        if engine is None:
            raise RecordError('the docker daemon does not answer')
        env = records.environment(local_repo, repo, environment_path(local_repo), 'repo2docker',
                                  records.tool_version('repo2docker'))
        return repo, env, records.machine(engine, host=host)
    env = records.environment(local_repo, repo, 'binder', 'host-conda', records.tool_version('conda'))
    return repo, env, records.machine('none', host=host)


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


def build(local_repo, runtime, *, name=None, remote=None, tag=None, records_dir=None, host=True):
    """Build the checkout's environment: a docker image with repo2docker, or ./.condaenv with conda.

    `runtime` is 'docker' or 'conda'. `name` is the images' name without a tag (default
    econ-ark/<folder name>). `remote` and `tag` are the catalog's, for the record; the
    remote defaults to the clone's origin. With `records_dir`, the record and the build's
    log are written there.

    Returns an Outcome. Nothing is built, and `problem` says why, when the facts a record
    needs cannot be read: the checkout is not a git repository, git records no file under
    the environment's directory, or docker does not answer. A docker build that fails has
    no image and so no record; its output is in `stdout`.
    """
    local_repo = Path(local_repo)
    try:
        repo, env, where = _facts(local_repo, runtime, remote, tag, host)
    except (RecordError, OSError) as e:
        return _refused(f'not built: {e}')
    if runtime == 'docker':
        named = image_name(name or default_name(local_repo), env)
        result = docker.build_docker(local_repo, named, labels={docker.COMMIT_LABEL: repo['commit'],
                                                               docker.ENVIRONMENT_LABEL: env['sha256']})
        image = docker.image_id(named) if result.returncode == 0 else None
        if image is None:
            return Outcome(result.returncode or 1, result.stdout, problem='the build made no image')
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


def execute(local_repo, runtime, script, *, arguments=(), image=None, name=None, remote=None, tag=None,
            records_dir=None, host=True, network=True, cpus=None, memory=None, threads=None, timeout=None,
            extensions=None):
    """Run `script` from the checkout in its built environment and record the run.

    `runtime` is 'docker' or 'conda'. For docker, the image is the one `build` named for
    the environment as it is now, or `image` (a name or an ID) when given; `network`,
    `cpus`, `memory`, `threads` and `timeout` limit the container and are docker's alone.
    `extensions` is a project's own part of the record, one object per namespace.

    Returns an Outcome. Nothing runs, and `problem` says why, when the facts cannot be
    read, when there is no image for the environment as it is, or when the image was
    built by this tool at another commit.
    """
    local_repo = Path(local_repo)
    if runtime != 'docker' and (not network or any(v is not None for v in (image, cpus, memory, threads, timeout))):
        return _refused('not run: image, network, cpus, memory, threads and timeout apply to docker only')
    try:
        repo, env, where = _facts(local_repo, runtime, remote, tag, host)
        conditions = {'network': bool(network)}
        if cpus is not None:
            conditions['cpus'] = cpus
        if memory is not None:
            conditions['memory_bytes'] = records.memory_bytes(memory)
        if threads is not None:
            conditions['thread_variables'] = {variable: str(threads) for variable in docker.THREAD_VARIABLES}
        if timeout is not None:
            conditions['wall_seconds'] = timeout
    except (RecordError, OSError) as e:
        return _refused(f'not run: {e}')

    image_block = None
    if runtime == 'docker':
        wanted = image or image_name(name or default_name(local_repo), env)
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
                                       cpus=cpus, memory=memory, threads=threads, timeout=timeout)
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

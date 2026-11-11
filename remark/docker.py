"""Building a REMARK's image with repo2docker and running its script in a container."""

from json import loads
from subprocess import run, PIPE, STDOUT, DEVNULL
from uuid import uuid4

from remark.records import by_signal, timed_run

THREAD_VARIABLES = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']
# What an image this tool built was built from, kept on the image itself; and the name
# it was built under, by which `clean` finds it after a later build has taken the name.
COMMIT_LABEL = 'org.econ-ark.remark.commit'
ENVIRONMENT_LABEL = 'org.econ-ark.remark.environment'
NAME_LABEL = 'org.econ-ark.remark.name'


def _docker(*args):
    return run(['docker', *args], stdout=PIPE, stderr=STDOUT, stdin=DEVNULL, encoding='utf-8', errors='replace')


def without_tag(image_name):
    """An image name without its tag: econ-ark/thing for econ-ark/thing:abc."""
    head, colon, tag = image_name.rpartition(':')
    return head if colon and '/' not in tag else image_name


def image_id(image):
    """The image's ID (sha256:...), its identity whatever its name does later; None if there is no such image."""
    proc = _docker('image', 'inspect', '--format', '{{.Id}}', image)
    return proc.stdout.strip() if proc.returncode == 0 else None


def image_label(image, label):
    """A label's value on the image, or None when the image does not carry it."""
    proc = _docker('image', 'inspect', '--format', '{{json .Config.Labels}}', image)
    try:
        return (loads(proc.stdout) or {}).get(label) if proc.returncode == 0 else None
    except ValueError:
        return None


def build_docker(local_repo, image_name, labels=None):
    """Build the checkout's image with repo2docker and name it `image_name`. Returns the build's Run.

    The build always runs. repo2docker skips it whenever an image of the name it is given
    exists, which would leave an image built from an older commit in place; so it is given
    a name no image has, and the result is then named `image_name`. Docker's layer cache
    keeps a build that changes nothing short. An image that held the name before stays,
    unnamed, for the records that refer to it by ID, until `clean_docker` or
    `docker image prune`.
    """
    repository = without_tag(image_name)
    scratch = f'{repository}:building-{uuid4().hex[:12]}'
    cmd = ['repo2docker', '--no-run', '--image-name', scratch]
    for label, value in {NAME_LABEL: repository, **(labels or {})}.items():
        cmd += ['--label', f'{label}={value}']
    result = timed_run([*cmd, local_repo.resolve()])
    if result.returncode == 0:
        named = _docker('tag', scratch, image_name)
        _docker('rmi', scratch)  # drops the scratch name only: the image keeps `image_name`
        if named.returncode != 0:
            result.returncode = named.returncode
            result.stdout += f'\ndocker tag {scratch} {image_name}:\n{named.stdout}'
    return result


def image_workdir(image):
    # The image's working directory. repo2docker sets it to the directory it copied the
    # repository into (its REPO_DIR, the image user's home by default), so mounting the
    # checkout there lays it exactly where reproduce.sh expects to run.
    proc = _docker('image', 'inspect', '--format', '{{.Config.WorkingDir}}', image)
    workdir = proc.stdout.strip() if proc.returncode == 0 else ''
    # An image with no working directory, or with '/', which cannot be a mount target,
    # gets the checkout at /remark.
    return workdir if workdir not in ('', '/') else '/remark'


def docker_run_prefix(local_repo, image, *, workdir, name, network=True, cpus=None,
                      memory=None, threads=None):
    """The `docker run` command up to and including the image, without the command to run.

    No -i/-t: docker refuses -t when stdin is not a terminal, so a terminal would make
    every run from CI, cron or a background job fail. The container is named, so that a
    timeout can kill it and its end can be read before it is removed. The checkout is
    mounted writable at `workdir`, where the script runs; existing reproduce.sh scripts
    write into it.
    """
    prefix = [
        'docker', 'run', '--init', '--name', name, '--entrypoint', '',
        '--mount', f'type=bind,source={local_repo.resolve()},target={workdir}',
        '--workdir', workdir,
    ]
    if not network:
        prefix += ['--network', 'none']
    if cpus is not None:
        prefix += ['--cpus', str(cpus)]
    if memory is not None:
        prefix += ['--memory', str(memory)]
    if threads is not None:
        for variable in THREAD_VARIABLES:
            prefix += ['--env', f'{variable}={threads}']
    prefix.append(image)
    return prefix


def conda_prefix(image):
    # repo2docker names the Python execution conda environment: "kernel" | "notebook"
    #   kernel is used if the notebook env has incompat libraries or Python version
    #   notebook should be used in other cases.
    # An image built without conda (from a plain Dockerfile) has neither: run the script
    # directly rather than fail on `conda env list`.
    proc = run(
        ['docker', 'run', '--rm', '--entrypoint', '', image, 'conda', 'env', 'list', '--json'],
        stdout=PIPE, stderr=PIPE, stdin=DEVNULL, encoding='utf-8', errors='replace'
    )
    try:
        envs = loads(proc.stdout)['envs'] if proc.returncode == 0 else []
    except (ValueError, KeyError):
        envs = []
    for prefix in ['/srv/conda/envs/kernel', '/srv/conda/envs/notebook']:
        if prefix in envs:
            return ['conda', 'run', '-p', prefix]
    return []


def execute_docker(local_repo, image, script, *, arguments=(), network=True, cpus=None,
                   memory=None, threads=None, timeout=None):
    """Run `bash <script> <arguments>` from the checkout at `local_repo` in `image`, a name or an ID.

    Returns the Run: the combined output, the container's exit code, the time around the
    script's own container, and how it ended. A run stopped at `timeout` seconds has its
    container killed and returns 124 ('timeout'); one the memory limit killed is 'oom'
    when docker reports it, which it does not always do (it missed 1 of 40 on docker
    29.1.3); an unreported one is a 'signal', SIGKILL, like any other kill.
    """
    workdir = image_workdir(image)
    name = f'remark-{local_repo.name.lower()}-{uuid4().hex[:8]}'
    prefix = docker_run_prefix(local_repo, image, workdir=workdir, name=name,
                               network=network, cpus=cpus, memory=memory, threads=threads)
    cmd = [*prefix, *conda_prefix(image), 'bash', script, *arguments]
    # Stopping the docker client leaves the container running: kill it by name.
    result = timed_run(cmd, timeout=timeout, on_timeout=lambda: _docker('kill', name))
    # An exit code of 137 alone cannot tell the memory limit from another kill: ask docker.
    state = _docker('inspect', '--format', '{{.State.ExitCode}} {{.State.OOMKilled}}', name)
    if state.returncode == 0 and result.termination != 'timeout':
        code, oom = state.stdout.split()
        result.returncode = int(code)
        result.termination, result.signal = ('oom', None) if oom == 'true' else by_signal(int(code))
    _docker('rm', '--force', name)
    return result


def clean_docker(name):
    """Remove every image named `name` (econ-ark/<name>) whatever its tag, and every image
    this tool built under that name, those a later build left unnamed included."""
    named = _docker('images', '--quiet', '--no-trunc', name)
    built = _docker('images', '--all', '--quiet', '--no-trunc', '--filter', f'label={NAME_LABEL}={name}')
    ids = sorted({i for listed in (named, built) if listed.returncode == 0 for i in listed.stdout.split()})
    if not ids:
        return named
    return run(['docker', 'rmi', '--force', *ids], encoding='utf-8')

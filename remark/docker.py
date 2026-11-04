"""Building a REMARK's image with repo2docker and running its script in a container."""

from json import loads
from subprocess import run, PIPE, STDOUT, DEVNULL, CompletedProcess, TimeoutExpired
from uuid import uuid4

THREAD_VARIABLES = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']


def build_docker(local_repo, image_name):
    cmd = ['repo2docker', '--no-run', '--image-name', image_name, local_repo.resolve()]
    return run(cmd, stdout=PIPE, stderr=STDOUT, encoding='utf-8')


def image_workdir(image_name):
    # The image's working directory. repo2docker sets it to the directory it copied the
    # repository into (its REPO_DIR, the image user's home by default), so mounting the
    # checkout there lays it exactly where reproduce.sh expects to run.
    proc = run(
        ['docker', 'image', 'inspect', '--format', '{{.Config.WorkingDir}}', image_name],
        stdout=PIPE, stderr=STDOUT, encoding='utf-8'
    )
    workdir = proc.stdout.strip() if proc.returncode == 0 else ''
    # An image with no working directory, or with '/', which cannot be a mount target,
    # gets the checkout at /remark.
    return workdir if workdir not in ('', '/') else '/remark'


def docker_run_prefix(local_repo, image_name, *, workdir, name, network=True, cpus=None,
                      memory=None, threads=None):
    """The `docker run` command up to and including the image, without the command to run.

    No -i/-t: docker refuses -t when stdin is not a terminal, so a terminal would make
    every run from CI, cron or a background job fail. The container is removed when it
    ends and is named, so that a timeout can kill it. The checkout is mounted writable at
    `workdir`, where the script runs; existing reproduce.sh scripts write into it.
    """
    prefix = [
        'docker', 'run', '--rm', '--init', '--name', name, '--entrypoint', '',
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
    prefix.append(image_name)
    return prefix


def conda_prefix(docker_prefix):
    # repo2docker names the Python execution conda environment: "kernel" | "notebook"
    #   kernel is used if the notebook env has incompat libraries or Python version
    #   notebook should be used in other cases.
    # An image built without conda (from a plain Dockerfile) has neither: run the script
    # directly rather than fail on `conda env list`.
    proc = run(
        [*docker_prefix, 'conda', 'env', 'list', '--json'],
        stdout=PIPE, stderr=PIPE, stdin=DEVNULL, encoding='utf-8'
    )
    try:
        envs = loads(proc.stdout)['envs'] if proc.returncode == 0 else []
    except (ValueError, KeyError):
        envs = []
    for prefix in ['/srv/conda/envs/kernel', '/srv/conda/envs/notebook']:
        if prefix in envs:
            return ['conda', 'run', '-p', prefix]
    return []


def execute_docker(local_repo, image_name, script, *, network=True, cpus=None,
                   memory=None, threads=None, timeout=None):
    """Run `bash <script>` from the checkout at `local_repo` in the image `image_name`.

    Returns a CompletedProcess with the combined output; a run stopped at `timeout`
    seconds has its container killed and returns 124.
    """
    workdir = image_workdir(image_name)
    name = f'remark-{local_repo.name.lower()}-{uuid4().hex[:8]}'
    prefix = docker_run_prefix(local_repo, image_name, workdir=workdir, name=name,
                               network=network, cpus=cpus, memory=memory, threads=threads)
    cmd = [*prefix, *conda_prefix(prefix), 'bash', script]
    try:
        return run(cmd, stdout=PIPE, stderr=STDOUT, stdin=DEVNULL, encoding='utf-8', timeout=timeout)
    except TimeoutExpired as e:
        # Stopping the docker client leaves the container running: kill it by name.
        run(['docker', 'kill', name], stdout=DEVNULL, stderr=DEVNULL)
        output = e.stdout or ''
        if isinstance(output, bytes):
            output = output.decode('utf-8', errors='replace')
        output += f'\nTIMEOUT: {script} was stopped after {timeout} s\n'
        return CompletedProcess(cmd, 124, stdout=output)


def clean_docker(image_name):
    cmd = ['docker', 'rmi', '--force', image_name]
    return run(cmd, encoding='utf-8')

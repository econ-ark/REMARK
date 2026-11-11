"""Building a REMARK's conda environment in its checkout and running its script there."""

from os import getenv
from subprocess import run

from remark.records import timed_run


def build_conda(local_repo):
    """Create or update ./.condaenv from binder/environment.yml. Returns the build's Run."""
    cmd = ['conda', 'env', 'update', '-f', 'binder/environment.yml', '--prefix', './.condaenv']
    result = timed_run(cmd, cwd=local_repo)
    if result.returncode == 0:
        with open(local_repo / '.condaenv' / '.gitignore', 'w') as f:
            f.write('*')
    return result


def execute_conda(local_repo, script, *, arguments=()):
    """Run the script in ./.condaenv from the checkout. Returns the Run."""
    cmd = ['conda', 'run', '-p', './.condaenv', getenv('SHELL', default='/bin/bash'), script, *arguments]
    return timed_run(cmd, cwd=local_repo)


def clean_conda(local_repo):
    cmd = ['conda', 'env', 'remove', '--prefix', './.condaenv', '--yes', '--quiet']
    return run(cmd, encoding='utf-8', cwd=local_repo)

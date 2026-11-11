"""Building a REMARK's conda environment in its checkout and running its script there."""

from os import getenv
from subprocess import run, PIPE, STDOUT


def build_conda(local_repo):
    cmd = ['conda', 'env', 'update', '-f', 'binder/environment.yml', '--prefix', './.condaenv']
    proc = run(cmd, stdout=PIPE, stderr=STDOUT, encoding='utf-8', cwd=local_repo)
    if proc.returncode == 0:
        with open(local_repo / '.condaenv' / '.gitignore', 'w') as f:
            f.write('*')
    return proc


def execute_conda(local_repo, script):
    cmd = ['conda', 'run', '-p', './.condaenv', getenv('SHELL', default='/bin/bash'), script]
    return run(cmd, stdout=PIPE, stderr=STDOUT, encoding='utf-8', cwd=local_repo)


def clean_conda(local_repo):
    cmd = ['conda', 'env', 'remove', '--prefix', './.condaenv', '--yes', '--quiet']
    return run(cmd, encoding='utf-8', cwd=local_repo)

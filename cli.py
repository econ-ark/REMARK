#!/usr/bin/env python

from csv import DictWriter, DictReader
from collections import defaultdict
from datetime import datetime, timezone
from itertools import tee, islice
from json import loads
from os import getenv
import re

from yaml import safe_load
from yaml.scanner import ScannerError

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from io import StringIO
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from subprocess import run, PIPE, STDOUT
from urllib.parse import urlsplit

@dataclass
class Metadata:
    remote: str
    local: Path
    yaml: dict
    image_name: str = None

    def __post_init__(self):
        self.image_name = f'econ-ark/{self.local.name.lower().replace(" ", "_")}'

    def flat(self):
        d = {**self.yaml, **asdict(self)}
        del d['yaml']
        return d


def parse_paths_from_standard(text):
    lines = text.splitlines()
    lines.pop(0) # skip the '.'

    d = Path()
    for prev, cur in zip(*(islice(it, i, None) for i, it in enumerate(tee(lines, 2)))):
        _, _, prev_part = prev.partition('--')
        _, _, cur_part = cur.partition('--')
        cur_part, prev_part = cur_part.strip(), prev_part.strip()

        prev_indent = len(re.findall(r'\s{4}|\|\s{3}', prev))
        cur_indent = len(re.findall(r'\s{4}|\|\s{3}', cur))

        if prev_indent == cur_indent:
            yield d / prev_part
        elif prev_indent > cur_indent:
            yield d / prev_part
            d = d.parent
        else:
            d = d / prev_part

    if prev_indent > cur_indent:
        yield d.parent / cur_part
    else:
        yield d / cur_part

def git_exists(local_repo_path):
    return Path(local_repo_path).joinpath('.git').exists()

def git_clone(local_repo_path, *, remote):
    return run(
        ['git', 'clone', '--depth', '1', '--single-branch', remote, local_repo_path]
    )

def git_pull(local_repo_path, *, remote_name=None):
    return run(['git', 'pull'], cwd=local_repo_path)

def git_update_remotes(local_repo_path, *, remote_dict):
    returns = {}
    for name, url in remote_dict.items():
        returns[name] = run(['git', 'remote', 'set-url', name, url], cwd=local_repo_path)
        if returns[name].returncode != 0:
            returns[name] = run(['git', 'remote', 'add', name, url], cwd=local_repo_path)
    return returns

def build_docker(local_repo, image_name):
    cmd = ['repo2docker', '--no-run', '--image-name', image_name, local_repo.resolve()]
    return run(cmd, stdout=PIPE, stderr=STDOUT, encoding='utf-8')

def execute_docker(local_repo, image_name):
    # repo2docker names the Python execution conda environment: "kernel" | "notebook"
    #   kernel is used if the notebook env has incompat libraries or Python version
    #   notebook should be used in other cases.
    docker_prefix = [
        'docker', 'run', '-it', '--entrypoint', '',
        '--mount', f'type=bind,source={local_repo.resolve()},target={getenv("HOME")}',
        image_name,
    ]

    envs_list_proc = run(
        [*docker_prefix, 'conda', 'env', 'list', '--json'],
        stdout=PIPE, stderr=STDOUT, encoding='utf-8'
    )
    envs = loads(envs_list_proc.stdout)['envs']

    priority = ['/srv/conda/envs/kernel', '/srv/conda/envs/notebook']
    for prefix in priority:
        if prefix in envs:
            cmd_prefix = ['conda', 'run', '-p', prefix]
            break
    else:
        cmd_prefix = []

    return run(
        [*docker_prefix, *cmd_prefix, 'bash', './reproduce.sh'],
        stdout=PIPE, stderr=STDOUT, encoding='utf-8'
    )

def clean_docker(image_name):
    cmd = ['docker', 'rmi', '--force', image_name]
    return run(cmd, encoding='utf-8')

def build_conda(local_repo):
    cmd = ['conda', 'env', 'update', '-f', 'binder/environment.yml', '--prefix', './condaenv']
    proc = run(cmd, stdout=PIPE, stderr=STDOUT, encoding='utf-8', cwd=local_repo)
    if proc.returncode == 0:
        with open(local_repo / 'condaenv' / '.gitignore', 'w') as f:
            f.write('*')
    return proc

def execute_conda(local_repo):
    cmd = ['conda', 'run', '-p', './condaenv', getenv('SHELL', default='/bin/bash'), 'reproduce.sh']
    return run(cmd, stdout=PIPE, stderr=STDOUT, encoding='utf-8', cwd=local_repo)

def clean_conda(local_repo):
    cmd = ['conda', 'env', 'remove', '--prefix', './condaenv', '--yes', '--quiet']
    return run(cmd, encoding='utf-8', cwd=local_repo)

if __name__ == '__main__':
    git_root = Path(__file__).parent
    remark_home = git_root / '_REMARK'
    repo_home = remark_home / 'repos'
    repo_home.mkdir(exist_ok=True, parents=True)


    with open(remark_home / '.gitignore', 'w') as f:
        f.write('**')

    metadata = {}
    for p in git_root.joinpath('REMARKs').glob('*.yml'):
        with open(p) as f:
            data = safe_load(f)
            metadata[p.stem] = Metadata(
                local=repo_home / data['name'],
                remote=data['remote'],
                yaml=data,
            )

    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='action')

    # pull/fetch
    pull_parser = subparsers.add_parser('pull')
    pull_group = pull_parser.add_mutually_exclusive_group(required=True)
    pull_group.add_argument('remark', default=[], nargs='*')
    pull_group.add_argument('--all', action='store_true')


    # lint
    lint_parser = subparsers.add_parser('lint')
    lint_group = lint_parser.add_mutually_exclusive_group(required=True)
    lint_group.add_argument('remark', nargs='*', default=[])
    lint_group.add_argument('--all', action='store_true')


    # build
    build_parser = subparsers.add_parser('build')
    build_parser.add_argument('type', choices=['docker', 'conda'])
    build_parser.add_argument('--jobs', '-J', default=4, type=int)

    build_group = build_parser.add_mutually_exclusive_group(required=True)
    build_group.add_argument('remark', default=[], nargs='*')
    build_group.add_argument('--all', action='store_true')


    # execute
    execute_parser = subparsers.add_parser('execute')
    execute_parser.add_argument('type', choices=['docker', 'conda'])
    execute_parser.add_argument('--jobs', '-J', default=4, type=int)

    execute_group = execute_parser.add_mutually_exclusive_group(required=True)
    execute_group.add_argument('remark', default=[], nargs='*')
    execute_group.add_argument('--all', action='store_true')

    # log
    log_parser = subparsers.add_parser('logs')

    # clean
    clean_parser = subparsers.add_parser('clean')
    clean_parser.add_argument('type', choices=['docker', 'conda'])

    clean_group = clean_parser.add_mutually_exclusive_group(required=True)
    clean_group.add_argument('remark', default=[], nargs='*')
    clean_group.add_argument('--all', action='store_true')

    args = parser.parse_args()

    if args.action == 'pull':
        to_pull = metadata.keys() if args.all else args.remark
        for name in to_pull:
            mdata = metadata[name]
            print(f'Updating {name} @ {mdata.local}')
            if git_exists(mdata.local):
                git_pull(mdata.local)
            else:
                git_clone(mdata.local, remote=mdata.remote)
            print('-' * 20, end='\n\n')

    elif args.action == 'lint':
        to_lint = metadata.keys() if args.all else args.remark
        with open(git_root / 'STANDARD.md') as f:
            standard = re.search(
                f'## the remark standard.*```(.*?)```',
                f.read(),
                flags=re.I | re.DOTALL
            ).group(1).strip()
        requirements = [*parse_paths_from_standard(standard)]
        for remark in to_lint:
            mdata = metadata[remark]
            messages = []

            for req in requirements:
                if not mdata.local.joinpath(req).exists():
                    messages.append(f'missing {req}')

            if messages:
                print(
                    f' {remark} '.center(50, '-'),
                    mdata.local,
                    *(f'- {m}' for m in messages),
                    sep='\n',
                    end='\n'*2,
                )

    elif args.action == 'build':
        report_dir = remark_home / 'logs' / 'build'
        report_dir.mkdir(exist_ok=True, parents=True)

        if args.remark:
            to_build = args.remark
        elif args.all:
            to_build = metadata.keys()

        with ThreadPoolExecutor(min(len(to_build), args.jobs)) as pool:
            def submitter(name):
                def _submitter(func, *args, **kwargs):
                    def wrapper(*args, **kwargs):
                        print(f'Building {name}')
                        return func(*args, **kwargs)
                    return pool.submit(wrapper, *args, **kwargs)
                return _submitter

            futures = {}
            for name in to_build:
                mdata = metadata[name]
                if args.type == 'docker':
                    fut = submitter(name)(build_docker, mdata.local, mdata.image_name)
                elif args.type == 'conda':
                    fut = submitter(name)(build_conda, mdata.local)
                futures[fut] = (mdata, args.type)


        for comp in as_completed(futures):
            mdata, build_type = futures[comp]
            proc = comp.result()

            remark_name = mdata.yaml['name']
            report_log_path = report_dir / f'{remark_name}_{build_type}.log'
            report_rc_path = report_dir / f'{remark_name}_{build_type}_rc.log'

            with open(report_log_path, 'w') as f:
                f.write(proc.stdout)
            with open(report_rc_path, 'w') as f:
                f.write(str(proc.returncode))
            print(f'{remark_name} → {proc.returncode}')
            if args.jobs == 1:
                print(proc.stdout)

    elif args.action == 'execute':
        report_dir = remark_home / 'logs' / 'execute'
        report_dir.mkdir(exist_ok=True, parents=True)
        if args.remark:
            to_build = args.remark
        elif args.all:
            to_build = metadata.keys()

        with ThreadPoolExecutor(min(len(to_build), args.jobs)) as pool:
            def submitter(name):
                def _submitter(func, *args, **kwargs):
                    def wrapper(*args, **kwargs):
                        print(f'Executing {name}')
                        return func(*args, **kwargs)
                    return pool.submit(wrapper, *args, **kwargs)
                return _submitter

            futures = {}
            for name in to_build:
                mdata = metadata[name]
                if args.type == 'docker':
                    fut = submitter(name)(execute_docker, mdata.local, mdata.image_name)
                elif args.type == 'conda':
                    fut = submitter(name)(execute_conda, mdata.local)
                futures[fut] = (mdata, args.type)


        for comp in as_completed(futures):
            mdata, build_type = futures[comp]
            proc = comp.result()

            remark_name = mdata.yaml['name']
            report_log_path = report_dir / f'{remark_name}_{build_type}.log'
            report_rc_path = report_dir / f'{remark_name}_{build_type}_rc.log'

            with open(report_log_path, 'w') as f:
                f.write(proc.stdout)
            with open(report_rc_path, 'w') as f:
                f.write(str(proc.returncode))
            print(f'{remark_name} → {proc.returncode}')
            if args.jobs == 1:
                print(proc.stdout)

    elif args.action == 'logs':
        results = defaultdict(lambda: defaultdict(dict))
        padding = max(len(k) for k in metadata.keys())
        for name in metadata.keys():
            report_dir = remark_home / 'logs'
            for log_file in sorted(report_dir.glob(f'*/*{name}*_rc.log')):
                name, log_type, _ = log_file.name.rsplit('_', maxsplit=2)
                results[log_file.parent.name][name][log_type] = log_file.read_text()

        for log_type, logs in results.items():
            padding = max(len(k) for k in logs)
            print(f'{log_type:-^{padding}}')
            for name, rc in logs.items():
                print(f'{name: <{padding}} = {rc}')

    elif args.action == 'clean':
        if args.remark:
            to_build = args.remark
        elif args.all:
            to_build = metadata.keys()

        for name in to_build:
            mdata = metadata[name]
            if args.type == 'docker':
                clean_docker(mdata.image_name)
            elif args.type == 'conda':
                clean_conda(mdata.local)


#!/usr/bin/env python

from csv import DictWriter, DictReader
from collections import defaultdict
from datetime import datetime, timezone
from itertools import tee, islice
from json import loads
from os import getenv
import re
from shutil import rmtree

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

def concrete_path(path):
    p = Path(path)
    if not p.exists():
        raise ValueError(f'{path} does not exist!')
    return p

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

def add_remark_arg_group(subparser, required=True):
    group = subparser.add_mutually_exclusive_group(required=required)
    group.add_argument(
        'remark',
        default=[],
        nargs='*',
        type=concrete_path,
        help='path(s) to REMARK metadata files (located under REMARKs/*.yml).'
    )
    group.add_argument('--all', action='store_true', help='pull/clone all REMARKs found in REMARKs/*.md')
    return group


def parse_paths_from_standard(text):
    lines = text.splitlines()
    lines.pop(0) # skip the '.'

    d = Path()
    for prev, cur in zip(*(islice(it, i, None) for i, it in enumerate(tee(lines, 2)))):
        if not cur.strip() or cur.startswith('#'):
            continue
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

def git_pull(local_repo_path):
    return run(['git', 'pull'], cwd=local_repo_path)

def git_checkout(local_repo_path, *, identifier):
    return run(['git', 'checkout', identifier], cwd=local_repo_path)

def build_docker(local_repo, image_name):
    cmd = ['repo2docker', '--no-run', '--image-name', image_name, local_repo.resolve()]
    return run(cmd, stdout=PIPE, stderr=STDOUT, encoding='utf-8')

def execute_docker(local_repo, image_name, script):
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
        [*docker_prefix, *cmd_prefix, 'bash', script],
        stdout=PIPE, stderr=STDOUT, encoding='utf-8'
    )

def clean_docker(image_name):
    cmd = ['docker', 'rmi', '--force', image_name]
    return run(cmd, encoding='utf-8')

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
            data['name'] = p.stem
            metadata[p.relative_to(git_root)] = Metadata(
                local=repo_home / data['name'],
                remote=data['remote'],
                yaml=data,
            )

    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='action')

    # pull/fetch
    pull_parser = subparsers.add_parser('pull', help='clone/pulls REMARK github repositories locally')
    add_remark_arg_group(pull_parser)

    # lint
    lint_parser = subparsers.add_parser('lint', help='check compatibility of REMARK repositories against STANDARD.md')
    add_remark_arg_group(lint_parser)
    lint_parser.add_argument('--include-optional', action='store_true', help='include optional files when checking against STANDARD.md')

    # build
    build_parser = subparsers.add_parser('build', help='build docker images/conda environments for REMARK repositories')
    build_parser.add_argument('type', choices=['docker', 'conda'])
    build_parser.add_argument('--jobs', '-J', default=4, type=int)
    add_remark_arg_group(build_parser)

    # execute
    execute_parser = subparsers.add_parser('execute', help='execute REMARK reproduce_min.sh (falling back to reproduce.sh) within their built environments')
    execute_parser.add_argument('type', choices=['docker', 'conda'],  help='execute within a built docker image or a conda environment')
    execute_parser.add_argument('--jobs', '-J', default=4, type=int, help='number of REMARKs to execute in parallel')
    execute_parser.add_argument('--no-min', action='store_true', help='ignore reproduce_min.sh')
    add_remark_arg_group(execute_parser)

    # log
    log_parser = subparsers.add_parser('logs', help='show most recent return codes from previous build/execute attempt')
    add_remark_arg_group(log_parser)

    # clean
    clean_parser = subparsers.add_parser('clean', help='remove build environments')
    clean_parser.add_argument('type', choices=['repo', 'docker', 'conda'])
    add_remark_arg_group(clean_parser)

    args = parser.parse_args()
    if args.action == 'pull':
        to_pull = metadata.keys() if args.all else args.remark
        for path in to_pull:
            mdata = metadata[path]
            print(f'Updating {path} @ {mdata.local}')
            if git_exists(mdata.local):
                git_pull(mdata.local)
            else:
                git_clone(mdata.local, remote=mdata.remote)

            if 'tag' in mdata.yaml:
                git_checkout(mdata.local, identifier=f'tags/{mdata.yaml["tag"]}')
            print('-' * 20, end='\n\n')

    elif args.action == 'lint':
        to_lint = metadata.keys() if args.all else args.remark
        with open(git_root / 'STANDARD.md') as f:
            standard = re.search(
                f'```\n\..*?```',
                f.read(),
                flags=re.I | re.DOTALL
            ).group(0).strip('`').strip()

        if args.include_optional:
            requirements = [
                p.with_name(p.name.rstrip('?')) for p in parse_paths_from_standard(standard)
            ]
        else:
            requirements = [
                p for p in parse_paths_from_standard(standard)
                if not p.name.endswith('?')
            ]

        for path in to_lint:
            mdata = metadata[path]
            messages = []

            for req in requirements:
                if not mdata.local.joinpath(req).exists():
                    messages.append(f'missing {req}')

            if messages:
                print(
                    f' {path} '.center(50, '-'),
                    mdata.local,
                    *(f'- {m}' for m in messages),
                    sep='\n',
                    end='\n'*2,
                )

    elif args.action == 'build':
        report_dir = remark_home / 'logs' / 'build'
        report_dir.mkdir(exist_ok=True, parents=True)
        to_build = metadata.keys() if args.all else args.remark

        with ThreadPoolExecutor(min(len(to_build), args.jobs)) as pool:
            def submitter(name):
                def _submitter(func, *args, **kwargs):
                    def wrapper(*args, **kwargs):
                        print(f'Building {name}')
                        return func(*args, **kwargs)
                    return pool.submit(wrapper, *args, **kwargs)
                return _submitter

            futures = {}
            for path in to_build:
                mdata = metadata[path]
                if args.type == 'docker':
                    fut = submitter(path)(build_docker, mdata.local, mdata.image_name)
                elif args.type == 'conda':
                    fut = submitter(path)(build_conda, mdata.local)
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
        to_execute = metadata.keys() if args.all else args.remark

        with ThreadPoolExecutor(min(len(to_execute), args.jobs)) as pool:
            def submitter(path):
                def _submitter(func, *args, **kwargs):
                    def wrapper(*args, **kwargs):
                        print(f'Executing {path}')
                        return func(*args, **kwargs)
                    return pool.submit(wrapper, *args, **kwargs)
                return _submitter

            futures = {}
            for path in to_execute:
                mdata = metadata[path]
                script = 'reproduce.sh'
                if (mdata.local / 'reproduce_min.sh').exists() and not args.no_min:
                    script = 'reproduce_min.sh'
                if args.type == 'docker':
                    fut = submitter(path)(
                        execute_docker,
                        local_repo=mdata.local,
                        image_name=mdata.image_name,
                        script=script
                    )
                elif args.type == 'conda':
                    fut = submitter(path)(
                        execute_conda,
                        local_repo=mdata.local,
                        script=script
                    )
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
        padding = max(len(str(k)) for k in metadata.keys())
        for path in metadata.keys():
            name = path.stem
            report_dir = remark_home / 'logs'
            for log_file in sorted(report_dir.glob(f'*/*{name}*_rc.log')):
                name, log_type, _ = log_file.name.rsplit('_', maxsplit=2)
                results[log_file.parent.name][name][log_type] = log_file.read_text().strip()

        for log_type, logs in results.items():
            padding = max(len(str(k)) for k in logs)
            print(f'{log_type:-^{padding}}')
            for name, rc in logs.items():
                print(f'{name: <{padding}} = {rc}')

    elif args.action == 'clean':
        to_clean = metadata.keys() if args.all else args.remark
        for name in to_clean:
            mdata = metadata[name]
            if args.type == 'repo':
                rmtree(mdata.local)
            elif args.type == 'docker':
                clean_docker(mdata.image_name)
            elif args.type == 'conda':
                clean_conda(mdata.local)


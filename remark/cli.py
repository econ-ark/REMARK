"""The command line: pull, lint, build, execute, logs and clean over the catalog.

Run from a REMARK checkout as `python cli.py <action> ...`, or as `remark <action> ...`
once the package is installed (from the checkout's root, or with --root).
"""

from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from shutil import rmtree
from subprocess import CalledProcessError

from remark.actions import build, execute
from remark.catalog import load_catalog
from remark.conda import clean_conda
from remark.docker import clean_docker
from remark.git import pull_entry
from remark.lint import lint_repo


def add_remark_arg_group(subparser, required=True):
    group = subparser.add_mutually_exclusive_group(required=required)
    group.add_argument(
        'remark',
        default=[],
        nargs='*',
        type=Path,
        help='path(s) to REMARK metadata files (located under REMARKs/*.yml).'
    )
    group.add_argument('--all', action='store_true', help='pull/clone all REMARKs found in REMARKs/*.yml')
    return group


def make_parser():
    parser = ArgumentParser(prog='remark')
    parser.add_argument('--root', type=Path, help='the REMARK checkout holding REMARKs/ (default: the current directory)')
    subparsers = parser.add_subparsers(dest='action', required=True)

    # pull/fetch
    pull_parser = subparsers.add_parser('pull', help='clone/pulls REMARK github repositories locally')
    add_remark_arg_group(pull_parser)

    # lint
    lint_parser = subparsers.add_parser('lint', help='check compatibility of REMARK repositories against STANDARD.md')
    add_remark_arg_group(lint_parser)
    lint_parser.add_argument('--include-optional', action='store_true', help='include optional files when checking against STANDARD.md')
    lint_parser.add_argument('--tier', type=int, choices=[1, 2, 3], default=2, help='REMARK tier to check against (1=Docker, 2=Reproducible, 3=Published). Default: 2')

    # build
    build_parser = subparsers.add_parser('build', help='build docker images/conda environments for REMARK repositories')
    build_parser.add_argument('type', choices=['docker', 'conda'])
    build_parser.add_argument('--jobs', '-J', default=4, type=int)
    build_parser.add_argument('--no-host', action='store_true', help="leave this machine's name out of the record")
    add_remark_arg_group(build_parser)

    # execute
    execute_parser = subparsers.add_parser('execute', help='execute REMARK reproduce_min.sh (falling back to reproduce.sh) within their built environments')
    execute_parser.add_argument('type', choices=['docker', 'conda'],  help='execute within a built docker image or a conda environment')
    execute_parser.add_argument('--jobs', '-J', default=4, type=int, help='number of REMARKs to execute in parallel')
    execute_parser.add_argument('--no-min', action='store_true', help='ignore reproduce_min.sh')
    execute_parser.add_argument('--no-host', action='store_true', help="leave this machine's name out of the record")
    caps = execute_parser.add_argument_group('docker only', 'limits on each container; none is set by default')
    caps.add_argument('--no-network', action='store_true', help='run with no network (docker run --network none)')
    caps.add_argument('--cpus', type=float, help='CPU limit for each container, such as 2')
    caps.add_argument('--memory', help='memory limit for each container, such as 8g')
    caps.add_argument('--threads', type=int, help='set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS')
    caps.add_argument('--timeout', type=float, help='stop a REMARK after this many seconds; its return code is then 124')
    add_remark_arg_group(execute_parser)

    # log
    log_parser = subparsers.add_parser('logs', help='show most recent return codes from previous build/execute attempt')
    add_remark_arg_group(log_parser)

    # clean
    clean_parser = subparsers.add_parser('clean', help='remove build environments')
    clean_parser.add_argument('type', choices=['repo', 'docker', 'conda'])
    add_remark_arg_group(clean_parser)
    return parser


def selected(parser, args, metadata, root):
    """The catalog keys the command names: all of them, or the files given."""
    if args.all:
        return list(metadata.keys())
    keys = []
    for p in args.remark:
        if not p.exists():
            parser.error(f'{p} does not exist!')
        try:
            key = p.resolve().relative_to(root)
        except ValueError:
            parser.error(f'{p} is not under {root}')
        if key not in metadata:
            parser.error(f'{p} is not a catalog entry (REMARKs/*.yml)')
        keys.append(key)
    return keys


def run_parallel(items, jobs, label, func):
    """Run func(item) for each item on `jobs` threads; yield (item, result) as each ends."""
    with ThreadPoolExecutor(max(1, min(len(items), jobs))) as pool:
        def wrapper(item):
            print(f'{label} {item[0]}')
            return func(item)
        futures = {pool.submit(wrapper, item): item for item in items}
        for comp in as_completed(futures):
            yield futures[comp], comp.result()


def report(report_dir, metadata_items, kind, jobs, action):
    """Write each REMARK's latest log and return code and say where its record is; exit
    non-zero naming any that failed or that left no record."""
    failed, unrecorded = [], []
    for (path, mdata), out in metadata_items:
        remark_name = mdata.yaml['name']
        with open(report_dir / f'{remark_name}_{kind}.log', 'w') as f:
            f.write(out.stdout)
        with open(report_dir / f'{remark_name}_{kind}_rc.log', 'w') as f:
            f.write(str(out.returncode))
        print(f'{remark_name} → {out.returncode}')
        if out.record_path:
            print(f'  record: {out.record_path}')
        if out.problem:
            print(f'  {out.problem}')
        if jobs == 1:
            print(out.stdout)
        if out.returncode != 0:
            failed.append(remark_name)
        elif out.record_path is None:
            unrecorded.append(remark_name)
    problems = []
    if failed:
        problems.append(f'{action} failed for: ' + ', '.join(sorted(failed)))
    if unrecorded:
        problems.append(f'{action} wrote no record for: ' + ', '.join(sorted(unrecorded)))
    if problems:
        raise SystemExit('; '.join(problems))


def main(argv=None, root=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    root = Path(args.root or root or Path.cwd()).resolve()
    if not root.joinpath('REMARKs').is_dir():
        parser.error(f'{root} has no REMARKs/ catalog; run from a REMARK checkout or pass --root')

    remark_home = root / '_REMARK'
    repo_home = remark_home / 'repos'
    records_home = remark_home / 'records'
    repo_home.mkdir(exist_ok=True, parents=True)
    with open(remark_home / '.gitignore', 'w') as f:
        f.write('**')

    metadata = load_catalog(root, repo_home)
    to_do = selected(parser, args, metadata, root)

    if args.action == 'pull':
        failed = []
        for path in to_do:
            mdata = metadata[path]
            tag = mdata.tag
            print(f'Updating {path} @ {mdata.local}' + (f' (tag {tag})' if tag else ''))
            try:
                pull_entry(mdata)
            except CalledProcessError as e:
                # Say so instead of silently leaving the clone on whatever it had:
                # lint/execute would otherwise report the default branch as the tag.
                print(f'ERROR: {path}: `{" ".join(map(str, e.cmd))}` exited {e.returncode}')
                failed.append(str(path))
            print('-' * 20, end='\n\n')
        if failed:
            raise SystemExit('pull failed for: ' + ', '.join(failed))

    elif args.action == 'lint':
        tier = args.tier
        failed = []
        for path in to_do:
            mdata = metadata[path]
            messages, warnings = lint_repo(mdata.local, tier=tier, include_optional=args.include_optional)
            if messages or warnings:
                print(f' {path} (Tier {tier}) '.center(50, '-'))
                print(mdata.local)
                if messages:
                    print('Errors:')
                    for m in messages:
                        print(f'  ❌ {m}')
                if warnings:
                    print('Warnings:')
                    for w in warnings:
                        print(f'  ⚠️  {w}')
                print()
            else:
                print(f'✅ {path} (Tier {tier}): All requirements met')
            if messages:
                failed.append(str(path))
        if failed:
            raise SystemExit(f'lint (tier {tier}) found errors in: ' + ', '.join(failed))

    elif args.action == 'build':
        report_dir = remark_home / 'logs' / 'build'
        report_dir.mkdir(exist_ok=True, parents=True)
        items = [(path, metadata[path]) for path in to_do]
        func = lambda item: build(
            item[1].local, args.type, name=item[1].image_name, remote=item[1].remote, tag=item[1].tag,
            records_dir=records_home / item[1].name, host=not args.no_host,
        )
        report(report_dir, run_parallel(items, args.jobs, 'Building', func), args.type, args.jobs, 'build')

    elif args.action == 'execute':
        if args.type == 'conda' and (args.no_network or any(
                v is not None for v in (args.cpus, args.memory, args.threads, args.timeout))):
            parser.error('--no-network, --cpus, --memory, --threads and --timeout apply to docker only')
        report_dir = remark_home / 'logs' / 'execute'
        report_dir.mkdir(exist_ok=True, parents=True)

        def script_for(mdata):
            if (mdata.local / 'reproduce_min.sh').exists() and not args.no_min:
                return 'reproduce_min.sh'
            return 'reproduce.sh'

        items = [(path, metadata[path]) for path in to_do]
        func = lambda item: execute(
            item[1].local, args.type, script_for(item[1]), name=item[1].image_name, remote=item[1].remote,
            tag=item[1].tag, records_dir=records_home / item[1].name, host=not args.no_host,
            network=not args.no_network, cpus=args.cpus, memory=args.memory, threads=args.threads,
            timeout=args.timeout,
        )
        report(report_dir, run_parallel(items, args.jobs, 'Executing', func), args.type, args.jobs, 'execute')

    elif args.action == 'logs':
        results = defaultdict(lambda: defaultdict(dict))
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
        for path in to_do:
            mdata = metadata[path]
            if args.type == 'repo':
                rmtree(mdata.local)
            elif args.type == 'docker':
                clean_docker(mdata.image_name)
            elif args.type == 'conda':
                clean_conda(mdata.local)


if __name__ == '__main__':
    main()

"""The status table: one row for every catalog entry, read from its repository as it is,
with the commit read, the newest commit on the remote, and what a reader found there.

Nothing is run and nothing is kept by hand. The frame is here: the catalog, the pin, the
commit read and the newest commit. What a row says beyond that comes from a reader chosen
by the kind of unit the repository declares in the front matter of its REMARK.md
(`kind: ...`, 'remark' when it declares none). A reader is a function

    reader(checkout: Path, entry: Metadata) -> {'summary': str, 'table': str, 'data': ...}

of which only `summary`, one line for the row, is required; `table` is the reader's own
Markdown for the unit, in whatever columns suit it, and `data` anything JSON can hold.
The reader for 'remark' is built in. Others are found among installed packages, under
the entry-point group `remark.status`, by the kind's name:

    [project.entry-points."remark.status"]
    comparison = "some_package.status:read"
"""

import json
from importlib import metadata as importlib_metadata
from pathlib import Path
from subprocess import run, PIPE, DEVNULL, CalledProcessError

from yaml import safe_load, YAMLError

from remark.git import pull_entry
from remark.lint import lint_repo

DEFAULT_KIND = 'remark'
ENTRY_POINT_GROUP = 'remark.status'


def front_matter(path):
    """The YAML between the first two `---` lines of a file, as a dict; {} when there is none."""
    try:
        lines = Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return {}
    marks = [i for i, line in enumerate(lines) if line.strip() == '---'][:2]
    if len(marks) < 2:
        return {}
    try:
        data = safe_load('\n'.join(lines[marks[0] + 1:marks[1]]))
    except YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def kind_of(checkout):
    """The kind of unit a checkout declares in REMARK.md's front matter, 'remark' when it declares none."""
    return str(front_matter(Path(checkout) / 'REMARK.md').get('kind') or DEFAULT_KIND)


def read_remark(checkout, entry):
    """The built-in reader: the highest tier whose required files the checkout holds."""
    met, missing = 0, []
    for tier in (1, 2, 3):
        errors, _ = lint_repo(Path(checkout), tier=tier)
        if errors:
            missing = errors
            break
        met = tier
    summary = f'meets tier {met}' if met else 'meets no tier'
    if missing and met < 3:
        summary += f'; for tier {met + 1}: ' + '; '.join(missing[:3]) + (' ...' if len(missing) > 3 else '')
    return {'summary': summary, 'data': {'tier': met, 'missing_for_next_tier': missing}}


def readers():
    """Every reader by kind: the built-in one, and those installed packages offer."""
    found = {}
    points = importlib_metadata.entry_points()
    group = points.select(group=ENTRY_POINT_GROUP) if hasattr(points, 'select') else points.get(ENTRY_POINT_GROUP, [])
    for point in group:
        found[point.name] = point.load
    found[DEFAULT_KIND] = lambda: read_remark  # the built-in kind is not replaced
    return found


def _git(*args, cwd=None):
    done = run(['git', *args], cwd=cwd, stdout=PIPE, stderr=DEVNULL, stdin=DEVNULL, encoding='utf-8')
    return done.stdout.strip() if done.returncode == 0 else None


def newest_commit(remote):
    """The commit at the head of the remote's default branch, or None when the remote does not answer."""
    out = _git('ls-remote', remote, 'HEAD')
    return out.split()[0] if out else None


def read_entry(entry, available=None):
    """One entry's row: pull it at its pin, then let the reader for its kind read the checkout.

    The row always comes back. `ok` is False, and `summary` says why, when the repository
    could not be pulled, no reader is installed for its kind, or the reader failed.
    """
    available = readers() if available is None else available
    row = {'name': entry.name, 'remote': entry.remote, 'private': entry.private, 'pin': entry.tag,
           'kind': None, 'commit': None, 'newest': None, 'ok': False, 'summary': ''}
    try:
        pull_entry(entry)
    except (CalledProcessError, OSError) as e:
        row['summary'] = f'not read: the pull failed ({e})'
        return row
    row['commit'] = _git('rev-parse', 'HEAD', cwd=entry.local)
    row['newest'] = newest_commit(entry.remote)
    row['kind'] = kind_of(entry.local)
    if row['kind'] not in available:
        row['summary'] = f'not read: no reader is installed for kind {row["kind"]!r}'
        return row
    try:
        found = available[row['kind']]()(entry.local, entry)
        row['summary'] = str(found['summary'])
    except Exception as e:  # a reader is another package's code: its failure is a row, not a crash
        row['summary'] = f'not read: the reader for {row["kind"]!r} failed ({type(e).__name__}: {e})'
        return row
    row['ok'] = True
    for key in ('table', 'data'):
        if found.get(key) is not None:
            row[key] = found[key]
    return row


def unread_row(entry):
    """The row of a private entry that was not asked for: that it exists, and nothing else."""
    return {'name': entry.name, 'remote': entry.remote, 'private': True, 'pin': entry.tag, 'kind': None,
            'commit': None, 'newest': None, 'ok': True, 'summary': 'private: not read'}


def _short(commit):
    return commit[:10] if commit else ''


def markdown(rows, stamp):
    """The table, and under it each unit's own table from its reader."""
    lines = ['# Status of the catalog', '', f'Read {stamp}. Nothing was run: each row is what the repository holds '
             'at the commit read.', '',
             '| entry | kind | pin | commit read | newest commit | private | status |',
             '|---|---|---|---|---|---|---|']
    for row in rows:
        newest = _short(row['newest'])
        if row['commit'] and row['newest'] and row['commit'] != row['newest']:
            newest += ' (newer)'
        cells = [row['name'], row['kind'] or '', row['pin'] or '', _short(row['commit']), newest,
                 'yes' if row['private'] else '', row['summary'].replace('|', '\\|').replace('\n', ' ')]
        lines.append('| ' + ' | '.join(cells) + ' |')
    for row in rows:
        if row.get('table'):
            lines += ['', f'## {row["name"]}', '', str(row['table']).strip()]
    return '\n'.join(lines) + '\n'


def write(rows, folder, stamp):
    """Write status.md and status.json into `folder`; return their paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    table, data = folder / 'status.md', folder / 'status.json'
    table.write_text(markdown(rows, stamp), encoding='utf-8')
    data.write_text(json.dumps({'read': stamp, 'entries': rows}, indent=1, default=str) + '\n', encoding='utf-8')
    return table, data

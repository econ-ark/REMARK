"""The status table: a row for every entry, read from its repository at the pin, nothing run."""

import json

import pytest

from conftest import commit_of, git
from remark import status
from remark.catalog import load_catalog
from remark.cli import main


def entry(catalog_root, name):
    entries = load_catalog(catalog_root, catalog_root / '_REMARK' / 'repos')
    return next(m for m in entries.values() if m.name == name)


@pytest.fixture
def comparison(tmp_path, catalog_root):
    """A second remote whose REMARK.md declares a kind of its own, catalogued as Compared."""
    repo = tmp_path / 'compared-origin'
    repo.mkdir()
    git('init', '--quiet', '--initial-branch', 'main', cwd=repo)
    git('config', 'user.email', 'test@example.org', cwd=repo)
    git('config', 'user.name', 'test', cwd=repo)
    (repo / 'REMARK.md').write_text('---\nkind: comparison\ntags:\n  - REMARK\n---\n\n# Compared\n')
    git('add', '-A', cwd=repo)
    git('commit', '--quiet', '-m', 'first', cwd=repo)
    (catalog_root / 'REMARKs' / 'Compared.yml').write_text(f'name: Compared\nremote: {repo.as_uri()}\ntitle: Compared\n')
    return repo


def test_front_matter_and_kind(tmp_path):
    assert status.kind_of(tmp_path) == 'remark'  # no REMARK.md
    (tmp_path / 'REMARK.md').write_text('# no front matter\n')
    assert status.front_matter(tmp_path / 'REMARK.md') == {} and status.kind_of(tmp_path) == 'remark'
    (tmp_path / 'REMARK.md').write_text('---\ntags: [REMARK]\nkind: comparison\n---\nbody\n---\nnot: metadata\n')
    assert status.front_matter(tmp_path / 'REMARK.md') == {'tags': ['REMARK'], 'kind': 'comparison'}
    assert status.kind_of(tmp_path) == 'comparison'
    (tmp_path / 'REMARK.md').write_text('---\n: not yaml: [\n---\n')
    assert status.kind_of(tmp_path) == 'remark'


def test_a_remark_is_read_at_its_pin(catalog_root, origin):
    row = status.read_entry(entry(catalog_root, 'Pinned'))
    assert row['ok'] and row['kind'] == 'remark' and row['pin'] == 'v1.0.0'
    assert row['commit'] == commit_of(origin, 'v1.0.0') and row['newest'] == commit_of(origin, 'main')
    assert row['summary'].startswith('meets no tier; for tier 1: ') and 'missing Dockerfile' in row['summary']
    assert row['data']['tier'] == 0 and 'missing Dockerfile' in row['data']['missing_for_next_tier']


def test_the_built_in_reader_reports_the_highest_tier(tmp_path):
    for name in ('Dockerfile', 'reproduce.sh', 'LICENSE'):
        (tmp_path / name).write_text('x\n')
    (tmp_path / 'binder').mkdir()
    (tmp_path / 'binder' / 'environment.yml').write_text('dependencies: []\n')
    (tmp_path / 'README.md').write_text('\n'.join(f'line {i}' for i in range(60)) + '\n')
    found = status.read_remark(tmp_path, None)
    assert found['data']['tier'] == 1 and found['summary'].startswith('meets tier 1; for tier 2: ')


def test_a_kind_is_read_by_its_reader(catalog_root, comparison):
    def read(checkout, unit):
        assert (checkout / 'REMARK.md').exists() and unit.name == 'Compared'
        return {'summary': '2 of 3 cells have a run of record', 'table': '| toolkit | status |\n|---|---|\n| a | ok |',
                'data': {'cells': 3}}

    row = status.read_entry(entry(catalog_root, 'Compared'), {'comparison': lambda: read})
    assert row['ok'] and row['kind'] == 'comparison' and row['summary'] == '2 of 3 cells have a run of record'
    assert row['data'] == {'cells': 3} and row['commit'] == row['newest'] == commit_of(comparison, 'main')


def test_what_cannot_be_read_is_a_row_that_says_why(catalog_root, comparison):
    compared = entry(catalog_root, 'Compared')
    row = status.read_entry(compared, {'remark': lambda: status.read_remark})
    assert not row['ok'] and row['summary'] == "not read: no reader is installed for kind 'comparison'"

    def broken(checkout, unit):
        raise KeyError('toolkits.yaml')

    row = status.read_entry(compared, {'comparison': lambda: broken})
    assert not row['ok'] and "the reader for 'comparison' failed (KeyError" in row['summary']

    pinned = entry(catalog_root, 'Pinned')
    pinned.yaml['tag'] = 'v9.9.9'
    row = status.read_entry(pinned)
    assert not row['ok'] and row['summary'].startswith('not read: the pull failed') and row['commit'] is None


def test_readers_come_from_installed_packages(monkeypatch):
    class Point:
        def __init__(self, name, target):
            self.name, self.load = name, lambda: target

    class Points:
        def select(self, group):
            assert group == 'remark.status'
            return [Point('comparison', 'the comparison reader'), Point('remark', 'an impostor')]

    monkeypatch.setattr(status.importlib_metadata, 'entry_points', lambda: Points())
    found = status.readers()
    assert found['comparison']() == 'the comparison reader'
    assert found['remark']() is status.read_remark  # the built-in kind is not replaced


def test_a_file_that_holds_private_results_says_so(catalog_root, origin, monkeypatch, capsys):
    monkeypatch.chdir(catalog_root)
    (catalog_root / 'REMARKs' / 'Secret.yml').write_text(
        f'name: Secret\nremote: {origin.as_uri()}\ntitle: Secret\nprivate: true\n')
    folder = catalog_root / '_REMARK' / 'status'
    main(['status', '--all'])  # the private entry is not read: nothing to warn of
    assert 'Not for publication' not in (folder / 'status.md').read_text()
    assert json.loads((folder / 'status.json').read_text())['private_entries_read'] == []
    assert 'not for publication' not in capsys.readouterr().out
    main(['status', '--all', '--private'])
    table = (folder / 'status.md').read_text()
    assert table.splitlines()[2] == ('> **Not for publication.** This file holds what was read from private '
                                     'repositories: Secret.')
    assert json.loads((folder / 'status.json').read_text())['private_entries_read'] == ['Secret']
    assert 'not for publication' in capsys.readouterr().out


def test_the_command_writes_a_row_for_every_entry(catalog_root, origin, comparison, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(catalog_root)
    nowhere = (tmp_path / 'not-reachable').as_uri()
    (catalog_root / 'REMARKs' / 'Hidden.yml').write_text(f'name: Hidden\nremote: {nowhere}\ntitle: Hidden\nprivate: true\n')

    def read(checkout, unit):
        return {'summary': 'read', 'table': '| cell | status |\n|---|---|\n| one | ok |'}

    monkeypatch.setattr(status, 'readers', lambda: {'remark': lambda: status.read_remark, 'comparison': lambda: read})
    main(['status', '--all'])
    out = capsys.readouterr().out
    folder = catalog_root / '_REMARK' / 'status'
    table, data = (folder / 'status.md').read_text(), json.loads((folder / 'status.json').read_text())
    assert [row['name'] for row in data['entries']] == ['Compared', 'Hidden', 'Pinned', 'Untagged']
    hidden = data['entries'][1]
    assert hidden['summary'] == 'private: not read' and hidden['commit'] is None and hidden['ok']
    pinned = commit_of(origin, 'v1.0.0')[:10]
    assert f'| Pinned | remark | v1.0.0 | {pinned} | {commit_of(origin, "main")[:10]} (newer) |  | meets no tier' in table
    assert '| Hidden |  |  |  |  | yes | private: not read |' in table
    assert '## Compared\n\n| cell | status |' in table and str(folder / 'status.md') in out

    # an entry that cannot be read has its row, and the command exits non-zero naming it
    with pytest.raises(SystemExit) as e:
        main(['status', '--all', '--private', '--output', str(tmp_path / 'elsewhere')])
    assert str(e.value.code) == 'status could not read: Hidden'
    rows = json.loads((tmp_path / 'elsewhere' / 'status.json').read_text())['entries']
    assert [row['ok'] for row in rows] == [True, False, True, True] and 'the pull failed' in rows[1]['summary']

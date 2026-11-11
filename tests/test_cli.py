"""The command line's exit codes and argument checks, on a catalog in a temporary folder."""

import subprocess
import sys
from pathlib import Path

import pytest

from conftest import commit_of, git
from remark.cli import main

REPO = Path(__file__).resolve().parent.parent


def test_pull_then_lint(catalog_root, origin, monkeypatch, capsys):
    monkeypatch.chdir(catalog_root)
    main(['pull', 'REMARKs/Pinned.yml'])
    clone = catalog_root / '_REMARK' / 'repos' / 'Pinned'
    assert git('rev-parse', 'HEAD', cwd=clone) == commit_of(origin, 'v1.0.0')
    # v1.0.0 has no Dockerfile: lint names it and exits non-zero
    with pytest.raises(SystemExit) as e:
        main(['lint', '--tier', '1', 'REMARKs/Pinned.yml'])
    assert 'Pinned.yml' in str(e.value.code)
    assert 'missing Dockerfile' in capsys.readouterr().out


def test_pull_with_a_missing_tag_exits_non_zero(catalog_root, monkeypatch):
    monkeypatch.chdir(catalog_root)
    entry = catalog_root / 'REMARKs' / 'Pinned.yml'
    entry.write_text(entry.read_text().replace('v1.0.0', 'v9.9.9'))
    with pytest.raises(SystemExit) as e:
        main(['pull', 'REMARKs/Pinned.yml'])
    assert 'pull failed for: REMARKs/Pinned.yml' in str(e.value.code)


def test_root_option_and_paths_from_elsewhere(catalog_root, origin, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(['--root', str(catalog_root), 'pull', str(catalog_root / 'REMARKs' / 'Untagged.yml')])
    clone = catalog_root / '_REMARK' / 'repos' / 'Untagged'
    assert git('rev-parse', 'HEAD', cwd=clone) == commit_of(origin, 'main')


def test_rejects_a_file_outside_the_catalog(catalog_root, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(catalog_root)
    stray = tmp_path / 'stray.yml'
    stray.write_text('name: x\n')
    with pytest.raises(SystemExit) as e:
        main(['pull', str(stray)])
    assert e.value.code == 2


def test_private_entries_are_left_out_of_all_unless_asked_for(catalog_root, origin, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(catalog_root)
    nowhere = (tmp_path / 'not-reachable').as_uri()
    (catalog_root / 'REMARKs' / 'Hidden.yml').write_text(f'name: Hidden\nremote: {nowhere}\ntitle: Hidden\nprivate: true\n')
    main(['pull', '--all'])  # the two public entries; the private one is not attempted
    assert 'Left out, marked private (--private includes them): Hidden' in capsys.readouterr().out
    repos = catalog_root / '_REMARK' / 'repos'
    assert (repos / 'Pinned').is_dir() and (repos / 'Untagged').is_dir() and not (repos / 'Hidden').exists()
    for asked in (['pull', '--all', '--private'], ['pull', 'REMARKs/Hidden.yml']):  # asked for: attempted, and it fails
        with pytest.raises(SystemExit) as e:
            main(asked)
        assert 'pull failed for: REMARKs/Hidden.yml' in str(e.value.code)
        assert 'marked private' in capsys.readouterr().out


def test_private_must_be_true_or_false(catalog_root, monkeypatch):
    monkeypatch.chdir(catalog_root)
    entry = catalog_root / 'REMARKs' / 'Untagged.yml'
    entry.write_text(entry.read_text() + 'private: for now\n')
    with pytest.raises(ValueError, match='`private` must be true or false'):
        main(['lint', '--all'])


def test_docker_only_limits_are_refused_for_conda(catalog_root, monkeypatch):
    monkeypatch.chdir(catalog_root)
    with pytest.raises(SystemExit) as e:
        main(['execute', '--cpus', '2', 'conda', 'REMARKs/Pinned.yml'])
    assert e.value.code == 2


def test_no_catalog_is_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(['logs', '--all'])
    assert e.value.code == 2


def test_cli_py_still_runs_from_a_checkout():
    # `python cli.py` from anywhere uses the checkout it sits in
    proc = subprocess.run([sys.executable, str(REPO / 'cli.py'), 'lint', '--help'],
                          capture_output=True, text=True, cwd='/')
    assert proc.returncode == 0 and '--tier' in proc.stdout

"""pull checks out the catalogued tag, on a fresh clone and on an existing one (#188)."""

import subprocess

import pytest

from conftest import commit_of, git
from remark.catalog import load_catalog
from remark.git import pull_entry


def entries(catalog_root):
    return load_catalog(catalog_root, catalog_root / '_REMARK' / 'repos')


def head(repo):
    return git('rev-parse', 'HEAD', cwd=repo)


def test_fresh_clone_lands_on_the_tag(catalog_root, origin):
    pinned = next(m for m in entries(catalog_root).values() if m.name == 'Pinned')
    pull_entry(pinned)
    assert head(pinned.local) == commit_of(origin, 'v1.0.0')
    assert not (pinned.local / 'Dockerfile').exists()


def test_pull_again_stays_on_the_tag(catalog_root, origin):
    pinned = next(m for m in entries(catalog_root).values() if m.name == 'Pinned')
    pull_entry(pinned)
    pull_entry(pinned)
    assert head(pinned.local) == commit_of(origin, 'v1.0.0')


def test_moving_the_tag_in_the_catalog_moves_the_clone(catalog_root, origin):
    pinned = next(m for m in entries(catalog_root).values() if m.name == 'Pinned')
    pull_entry(pinned)
    pinned.yaml['tag'] = 'v1.1.0'
    pull_entry(pinned)
    assert head(pinned.local) == commit_of(origin, 'v1.1.0')
    assert (pinned.local / 'Dockerfile').exists()


def test_a_clone_left_on_the_default_branch_moves_to_the_tag(catalog_root, origin):
    # what the code before #188 left behind: a shallow clone of main
    pinned = next(m for m in entries(catalog_root).values() if m.name == 'Pinned')
    subprocess.run(['git', 'clone', '--quiet', '--depth', '1', '--single-branch', origin.as_uri(), str(pinned.local)], check=True)
    assert head(pinned.local) == commit_of(origin, 'main')
    pull_entry(pinned)
    assert head(pinned.local) == commit_of(origin, 'v1.0.0')


def test_an_untagged_entry_follows_the_default_branch(catalog_root, origin):
    untagged = next(m for m in entries(catalog_root).values() if m.name == 'Untagged')
    pull_entry(untagged)
    assert head(untagged.local) == commit_of(origin, 'main')
    pull_entry(untagged)
    assert head(untagged.local) == commit_of(origin, 'main')


def test_a_missing_tag_raises(catalog_root):
    pinned = next(m for m in entries(catalog_root).values() if m.name == 'Pinned')
    pinned.yaml['tag'] = 'v9.9.9'
    with pytest.raises(subprocess.CalledProcessError):
        pull_entry(pinned)

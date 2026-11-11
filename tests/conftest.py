import shutil
import subprocess
from pathlib import Path

import pytest


def git(*args, cwd):
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def origin(tmp_path):
    """A repository standing in for a REMARK's remote: v1.0.0 lacks Dockerfile,
    v1.1.0 has it, and main has moved on past both."""
    repo = tmp_path / 'origin'
    repo.mkdir()
    git('init', '--quiet', '--initial-branch', 'main', cwd=repo)
    git('config', 'user.email', 'test@example.org', cwd=repo)
    git('config', 'user.name', 'test', cwd=repo)
    (repo / 'README.md').write_text('first\n')
    git('add', '-A', cwd=repo)
    git('commit', '--quiet', '-m', 'first', cwd=repo)
    git('tag', '-a', 'v1.0.0', '-m', 'v1.0.0', cwd=repo)
    (repo / 'Dockerfile').write_text('FROM scratch\n')
    git('add', '-A', cwd=repo)
    git('commit', '--quiet', '-m', 'add Dockerfile', cwd=repo)
    git('tag', '-a', 'v1.1.0', '-m', 'v1.1.0', cwd=repo)
    (repo / 'later.txt').write_text('after the tags\n')
    git('add', '-A', cwd=repo)
    git('commit', '--quiet', '-m', 'later', cwd=repo)
    return repo


def commit_of(repo, ref):
    return git('rev-parse', f'{ref}^{{commit}}', cwd=repo)


@pytest.fixture
def catalog_root(tmp_path, origin):
    """A REMARK checkout holding a catalog of two entries on `origin`: one pinned to
    v1.0.0, one untagged."""
    root = tmp_path / 'REMARK'
    (root / 'REMARKs').mkdir(parents=True)
    url = origin.as_uri()  # file://, so that shallow clones work
    (root / 'REMARKs' / 'Pinned.yml').write_text(f'name: Pinned\nremote: {url}\ntitle: Pinned\ntag: v1.0.0\n')
    (root / 'REMARKs' / 'Untagged.yml').write_text(f'name: Untagged\nremote: {url}\ntitle: Untagged\n')
    return root


def docker_available():
    if shutil.which('docker') is None:
        return False
    return subprocess.run(['docker', 'info'], capture_output=True).returncode == 0


def repo2docker_available():
    return docker_available() and shutil.which('repo2docker') is not None

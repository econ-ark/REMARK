"""Cloning and updating a catalog entry's repository at its catalogued tag."""

from pathlib import Path
from subprocess import run


def git_exists(local_repo_path):
    return Path(local_repo_path).joinpath('.git').exists()


def git_clone(local_repo_path, *, remote, tag=None):
    # A shallow single-branch clone is all lint/build/execute need, but it carries
    # no tags, so a later `git checkout tags/<tag>` on it fails. When the catalog
    # pins a tag, clone the tag itself: `--branch` accepts a tag and leaves HEAD
    # detached at its commit.
    cmd = ['git', 'clone', '--depth', '1', '--single-branch']
    if tag:
        cmd += ['--branch', tag]
    return run([*cmd, remote, local_repo_path], check=True)


def git_pull(local_repo_path):
    return run(['git', 'pull'], cwd=local_repo_path, check=True)


def git_fetch_tag(local_repo_path, *, tag):
    # Bring (or refresh) one tag into an existing shallow clone, whichever commit
    # it was cloned at.
    refspec = f'+refs/tags/{tag}:refs/tags/{tag}'
    return run(['git', 'fetch', '--depth', '1', 'origin', refspec], cwd=local_repo_path, check=True)


def git_checkout(local_repo_path, *, identifier):
    return run(['git', 'checkout', '--quiet', identifier], cwd=local_repo_path, check=True)


def pull_entry(mdata):
    """Clone or update one entry's repository and check out its catalogued tag.

    An entry without a tag is cloned or pulled on its default branch. A failing git
    step raises subprocess.CalledProcessError, never leaving the clone silently on
    whatever it had.
    """
    tag = mdata.tag
    if git_exists(mdata.local):
        if tag:
            git_fetch_tag(mdata.local, tag=tag)
        else:
            git_pull(mdata.local)
    else:
        git_clone(mdata.local, remote=mdata.remote, tag=tag)
    if tag:
        git_checkout(mdata.local, identifier=f'tags/{tag}')

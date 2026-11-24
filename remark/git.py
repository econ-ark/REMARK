"""Cloning and updating a catalog entry's repository at its catalogued tag, and reading
its remote as someone without credentials would."""

import os
from pathlib import Path
from subprocess import TimeoutExpired, run
from tempfile import TemporaryDirectory


# git as a stranger runs it: no global or system configuration (so no credential helper, no
# stored token and no url rewriting), no prompt for a username, ssh without a passphrase
# prompt, and none of the transports that run a command (`ext::`, `fd::`).
ANONYMOUS_GIT = {
    'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_CONFIG_NOSYSTEM': '1',
    'GIT_TERMINAL_PROMPT': '0',
    'GIT_SSH_COMMAND': 'ssh -o BatchMode=yes',
    'GIT_ALLOW_PROTOCOL': 'https:http:git:ssh:file',
}


def read_remote(remote, *, tag=None, timeout=60):
    """What someone without credentials finds at `remote`: None when it can be read and holds
    `tag` (if one is given), or else a line saying what is wrong.

    econ-ark.org's build clones every catalog entry not marked private this way, at its
    default branch, and fails if one cannot be read; `pull` also needs the tag. One
    `git ls-remote` answers both, without downloading the repository.
    """
    if remote.startswith('-'):
        return f'{remote!r} is not a remote'  # git would read it as an option
    patterns = ['HEAD'] + ([f'refs/tags/{tag}'] if tag else [])
    # run outside any repository, so that no repository's configuration applies either
    with TemporaryDirectory() as nowhere:
        try:
            result = run(['git', 'ls-remote', remote, *patterns], cwd=nowhere,
                         env={**os.environ, **ANONYMOUS_GIT}, capture_output=True, text=True, timeout=timeout)
        except TimeoutExpired:
            return f'no answer from {remote} in {timeout} s'
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        return (f'{remote} cannot be read without credentials: '
                + (lines[-1] if lines else f'git ls-remote exited {result.returncode}'))
    refs = {line.split('\t', 1)[1] for line in result.stdout.splitlines() if '\t' in line}
    if tag and f'refs/tags/{tag}' not in refs:
        return f'{remote} has no tag {tag}'
    return None


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

"""REMARK's command-line tool as a package: the catalog, git, lint, docker and conda steps
that `python cli.py` (or the installed `remark` command) runs, importable on their own.

    from remark.catalog import load_catalog
    from remark.git import pull_entry
    from remark.lint import lint_repo
    from remark.docker import build_docker, execute_docker
"""

__version__ = '0.1.0'

"""REMARK's command-line tool as a package: the catalog, git, lint, docker and conda steps
that `python cli.py` (or the installed `remark` command) runs, importable on their own.

    from remark.catalog import load_catalog
    from remark.git import pull_entry
    from remark.lint import lint_repo
    from remark.actions import build, execute       # each step with its record
    from remark.records import repository, environment, machine, validate
"""

__version__ = '0.2.0'

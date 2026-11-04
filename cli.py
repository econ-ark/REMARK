#!/usr/bin/env python
"""REMARK's command-line tool, run from a checkout: python cli.py <action> ...

The code is the `remark` package beside this file; `pip install .` (or
`pip install git+https://github.com/econ-ark/REMARK@<tag>`) installs it with a
`remark` command, and its steps can be imported (remark.git, remark.lint,
remark.docker, ...). See CLI-GUIDE.md.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from remark.cli import main  # noqa: E402

if __name__ == '__main__':
    main(root=ROOT)

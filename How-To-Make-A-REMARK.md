# How to Use REMARK

This document contains technical instructions for how to add new projects to the REMARK
archive, and how to work with and maintain the project repository itself.

## For Authors

Each project lives in its own repository. To make a new REMARK, you can start with a skeleton
[REMARK-starter-example](https://github.com/econ-ark/REMARK-starter-example) and add to it,
or from an example of a complete project using the toolkit, [BufferStockTheory](https://github.com/econ-ark/BufferStockTheory), whose content (code, text, figures, etc) you can replace with
your own.

REMARKs should adhere to the [REMARK Standard](https://github.com/econ-ark/REMARK/blob/master/STANDARD.md).

## For Editors

The REMARK catalog and Econ-ARK website configuration will be maintained by Editors.

Editorial guidelines are [here](https://github.com/econ-ark/REMARK/blob/master/EDITORIAL.md).

## For Maintainers

**Command Line Interface** `cli.py`

`cli.py` is an automated tool that facilitates:
- cloning of REMARK repositories
- linting (detection of missing files from a given REMARK)
- building conda environments/docker images
    - uses `conda`/`repo2docker` under the hood
- executing `reproduce.sh` scripts within the built environments.

All artifacts generated by `cli.py` are stored in a newly created `_REMARK` folder.

1. Once you clone a REMARK you'll be able to find its contents inside of `_REMARK/repos/…`
2. Once you build/execute a REMARK you'll be able to find a corresponding log
file from that process inside of `_REMARK/logs/…`

`cli.py` has built-in parallelization specified by the `-J` flag for many actions.

### Requirements

- python 3.9 or newer.
- contents `requirements.txt`

### Action

**Clone/Pull**

pulling REMARKs (these are populated in the  `_REMARKS` folder)

```bash
python cli.py pull --all         # git clone all REMARKS
python cli.py pull {remark_name} # git clone one or more REMARK(s)
```

**Lint**

Shows what files are missing from given REMARK(s). The linter uses the
file-tree print out from STANDARD.md and compares it to the current files found
in the currently cloned REMARK(s).

```bash
python cli.py lint --all # detect missing files from all REMARKs
python cli.py lint {remark_name} # detect missing files from one or more REMARK(s)
```

**Build**

Building conda environments and/or docker images.

```bash
python cli.py build conda --all          # build conda environments for all REMARKs (stored as a `condaenv` folder inside the cloned REMARK repo)
python cli.py build docker --all         # build docker images for all REMARKs (stored as a `condaenv` folder inside the cloned REMARK repo)
python cli.py build conda {remark_name}  # build conda environments for one or more REMARK(s)
python cli.py build docker {remark_name} # build docker image(s) for one or more REMARK(s)
```

The primary difference between `conda` and `docker` for builds are that `docker` will be more flexible for multilanguage REMARKs. It leverages 
repo2docker (same tool that mybinder uses) to create docker images from repositories.

**Execute**

Automated execution within built conda environments/docker containers.

```bash
python cli.py execute conda --all          # execute reproduce.sh via conda for all REMARKs
python cli.py execute docker --all         # execute reproduce.sh via docker for all REMARKs
python cli.py execute conda {remark_name}  # execute reproduce.sh via conda for one or more REMARK(s)
python cli.py execute docker {remark_name} # execute reproduce.sh via docker for one or more REMARK(s)
```

*Both the build and execute subcommands have an optional --jobs argument to
specify the number of jobs to run in parallel when building/executing.*

**Logs/Summarize**

```bash
python cli.py logs # view most recent logs for all previous building/executing commands
```

**Clean/Remove**

```bash
python cli.py clean conda --all          # remove all built conda environments
python cli.py clean docker --all         # remove all build docker images
python cli.py clean conda {remark_name}  # remove conda environment(s) from specified REMARK(s)
python cli.py clean docker {remark_name} # remove docker images built from specified REMARK(s)
```
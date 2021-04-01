# REMARK Guidelines

A REMARK refers to an open git repository that is indexed in this repository with appropriate metadata.

## Submitting a REMARK

To index the repository as a REMARK, please file a pull request in [this repository](https://github.com/econ-ark/REMARK).

The PR should add a link to the repository to the Catalog (currently, in the README).

## The REMARK Standard

The REMARK's repository must:
 1. Have a [tagged release](https://docs.github.com/en/github/administering-a-repository/managing-releases-in-a-repository), the last commit before including it as a REMARK should be tagged with a 1.0 release.
 2. In that repository at that release, there must be:
   - In either the top-level directory or a `binder/` directory, either:
     - installation files for `pip`:
       - a `runtime.txt` containing the name of a python version, e.g. `python-3.9.0`
       - a `requirements.txt` file with pinned dependencies (such as created by the command `pip freeze > requirements.txt`), or...
     - installation files for conda:
      - an `environment.yml` file with pinned dependencies
   - A `reproduce.sh` script that
     - Installs the requirements
     - Runs and reproduces all the results
3. Include a valid CITATION.cff document with bibliographic
metadata for the repository.
     
It is **strongly recommended** to include:
  - If reproduce.sh takes longer than a few minutes, a `reproduce_min.sh` that generates some interesting subset of results within a few minutes
  - A Jupyter notebook that exposits the material being reproduced.

A maximalist REMARK (the extra stuff is completely optional) includes:
  - A reproduce_text-etc.sh that generates the text
  - A dashboard that creates interactive versions of interesting figures

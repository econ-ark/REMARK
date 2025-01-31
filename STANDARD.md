# REMARK Guidelines

A REMARK refers to an open git repository that is indexed in this repository with appropriate metadata.

## Preparing your Repository

### Minimal Repository Structure

```
.
|-- reproduce.sh
|-- reproduce_min.sh?
|-- CITATION.cff
`-- binder/
    |-- environment.yml
#   |-- {optional binder files}?

# ? indicates an optional requirement
# {optional binder files} are neither prohibited nor required but allow one to
#   further customize the execution environment of their repository. For a list
#   of full files you can place in this folder see:
#   https://mybinder.readthedocs.io/en/latest/using/config_files.html.
```

1. Have a [tagged release](https://docs.github.com/en/github/administering-a-repository/managing-releases-in-a-repository).
    - If you make changes to your work, please release a new version/tag.
    - Commonly, tags will follow [semantic versioning](https://en.wikipedia.org/wiki/Software_versioning#Semantic_versioning) having a 3 part numeric identifier `v{major}.{minor}.{patch}` where:
        - major version changes represent large refactors, new features, and/or new analyses.
        - minor version changes represent small refactoring efforts and small added functionality.
        - patch changes represent small fixes made in the codebase, but no change in overall functionality.
2. In that repository at that release, there must be:
    - A `binder/` directory containing configuration files to recreate your execution environment:
      - An `environment.yml` file with appropriately pinned dependencies.
      - Optionally, you may include any other files as part of the [binder configuration specification](https://mybinder.readthedocs.io/en/latest/using/config_files.html)
    - A `reproduce.sh` script that:
      - Runs and reproduces all the results. If this script fails, your reproduction is assumed to not have worked.
    - Optionally: `reproduce_min.sh` 
        - You may include this file if your full `reproduce.sh` takes ≥5 minutes to run on your local machine.
3. Include a valid [CITATION.cff](https://citation-file-format.github.io/) document with bibliographic metadata for the repository.
    - To create a CITATION.cff file, you may use the [citation file format initializer](https://citation-file-format.github.io/cff-initializer-javascript/#/start)

## Submitting a REMARK

To index your repository as a REMARK, please file a Pull Request in [this repository](https://github.com/econ-ark/REMARK).
Your Pull Request should either create a new catalog entry or increment the existing version entry for your REMARK.

Once you open a Pull Request, the Econ-ARK team will review your submission to ensure that your `reproduce.sh` script
is able to run in the provided environment. If you would like to test this process on your own, you may clone this 
repository, update the cataloged version, and build/execute your REMARK using the provided `cli.py` script.


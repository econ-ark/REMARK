# REMARK Guidelines

A REMARK refers to an open git repository that is indexed in this repository with appropriate metadata.

## Preparing your Repository

### Minimal Repository Structure

```bash
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
      - A `binder/environment.yml` file that can be used non-interactively to create an environment in which `reproduce.sh` runs successfully. For compatibility with the REMARK tooling (which currently builds environments with `conda env update -f binder/environment.yml`), this file MUST exist, but it may be either:
        - a fully specified conda environment with appropriately pinned dependencies; or
        - a minimal conda environment that installs your chosen environment manager (for example, `uv`, `poetry`, or `pip-tools`) and any tools it needs to install dependencies from a separate, pinned lockfile (for example, `uv.lock`, `poetry.lock`, or a compiled `requirements.txt`).
      - You are encouraged to maintain your primary environment configuration in the format of your chosen tool (for example, `pyproject.toml`/`uv.lock`, `pyproject.toml`/`poetry.lock`, or `requirements.in`/`requirements.txt`). In this case, `binder/environment.yml` serves as an adapter that ensures your tool is available inside the Binder/REMARK execution environment.
      - Optionally, you may include any other files as part of the [binder configuration specification](https://mybinder.readthedocs.io/en/latest/using/config_files.html)
    - A `reproduce.sh` script that:
      - Runs and reproduces all the results. If this script fails, your reproduction is assumed to not have worked.
    - Optionally: `reproduce_min.sh`
        - You may include this file if your full `reproduce.sh` takes ≥5 minutes to run on your local machine.
3. Include a valid [CITATION.cff](https://citation-file-format.github.io/) document with bibliographic metadata for the repository.
    - To create a CITATION.cff file, you may use the [citation file format initializer](https://citation-file-format.github.io/cff-initializer-javascript/#/start)
4. For "Published" REMARKs (ready for permanent archival and citation), additional requirements apply:
    - **Zenodo DOI Required**: Obtain a permanent Zenodo DOI for your repository
    - **Specific Git Tag Required**: The Zenodo archive must correspond to a specific git tag (e.g., `v1.0.0`)
    - **Version Verification**: The tag provides cryptographic proof that the econ-ark fork matches your Zenodo archive exactly
    
    **Process for Published REMARKs**:
    1. Create a git tag for the version to archive: `git tag -a v1.0.0 -m "Published version"`
    2. Push the tag to GitHub: `git push origin v1.0.0`
    3. Enable Zenodo-GitHub integration (see [GitHub-Zenodo guide](https://guides.github.com/activities/citable-code/))
    4. Create a GitHub release from that tag, which triggers Zenodo archival
    5. Obtain the Zenodo DOI from the archived version
    6. Add the DOI to your CITATION.cff: `doi: 10.5281/zenodo.XXXXXX`
    7. When submitting to econ-ark/REMARK, specify the exact git tag that was archived
    
    **Why This Matters**:
    - The git tag creates an immutable snapshot at a specific commit
    - The commit SHA-1 hash provides cryptographic verification
    - Anyone can verify versions match by comparing commit hashes
    - This guarantees: econ-ark fork = Zenodo archive = your tagged version

## Submitting a REMARK

To index your repository as a REMARK, please file a Pull Request in [this repository](https://github.com/econ-ark/REMARK).
Your Pull Request should either create a new catalog entry or increment the existing version entry for your REMARK.

Once you open a Pull Request, the Econ-ARK team will review your submission to ensure that your `reproduce.sh` script
is able to run in the provided environment. If you would like to test this process on your own, you may clone this
repository, update the cataloged version, and build/execute your REMARK using the provided `cli.py` script.

# REMARK Guidelines

A REMARK refers to an open git repository that is indexed in this repository with appropriate metadata.

REMARKs are organized into three tiers based on their level of reproducibility and archival readiness:

- **Tier 1 (Docker REMARK)**: Minimal reproducibility via Docker containerization
- **Tier 2 (Reproducible REMARK)**: Enhanced reproducibility with comprehensive documentation
- **Tier 3 (Published REMARK)**: Publication-ready with permanent archival (Zenodo DOI)

## Preparing your Repository

### Base Requirements (All Tiers)

All REMARKs must meet these fundamental requirements:

1. **Have a [tagged release](https://docs.github.com/en/github/administering-a-repository/managing-releases-in-a-repository).**
    - If you make changes to your work, please release a new version/tag.
    - Commonly, tags will follow [semantic versioning](https://en.wikipedia.org/wiki/Software_versioning#Semantic_versioning) having a 3 part numeric identifier `v{major}.{minor}.{patch}` where:
        - major version changes represent large refactors, new features, and/or new analyses.
        - minor version changes represent small refactoring efforts and small added functionality.
        - patch changes represent small fixes made in the codebase, but no change in overall functionality.

2. **Required Files (All Tiers):**
    - **`Dockerfile`**: Must be present in the repository root. This enables containerized execution and ensures maximum portability across different computing environments. The Dockerfile should be compatible with [repo2docker](https://repo2docker.readthedocs.io/) or follow standard Docker practices.
    - **`reproduce.sh`**: A script that runs and reproduces all the results. If this script fails, your reproduction is assumed to not have worked.
    - **`README.md`**: A comprehensive README file documenting your REMARK. See tier-specific requirements for minimum line counts.
    - **`LICENSE`**: A license file specifying the terms under which your code and content are distributed.
    - **`binder/environment.yml`**: A conda environment file that can be used non-interactively to create an environment in which `reproduce.sh` runs successfully. For compatibility with the REMARK tooling (which currently builds environments with `conda env update -f binder/environment.yml`), this file MUST exist, but it may be either:
        - a fully specified conda environment with appropriately pinned dependencies; or
        - a minimal conda environment that installs your chosen environment manager (for example, `uv`, `poetry`, or `pip-tools`) and any tools it needs to install dependencies from a separate, pinned lockfile (for example, `uv.lock`, `poetry.lock`, or a compiled `requirements.txt`).
      - You are encouraged to maintain your primary environment configuration in the format of your chosen tool (for example, `pyproject.toml`/`uv.lock`, `pyproject.toml`/`poetry.lock`, or `requirements.in`/`requirements.txt`). In this case, `binder/environment.yml` serves as an adapter that ensures your tool is available inside the Binder/REMARK execution environment.
      - Optionally, you may include any other files as part of the [binder configuration specification](https://mybinder.readthedocs.io/en/latest/using/config_files.html)

3. **Optional Files:**
    - **`reproduce_min.sh`**: You may include this file if your full `reproduce.sh` takes ≥5 minutes to run on your local machine. This provides a quick validation path for testing.

### Minimal Repository Structure

```bash
.
|-- Dockerfile
|-- reproduce.sh
|-- reproduce_min.sh?      # Optional
|-- README.md
|-- LICENSE
|-- CITATION.cff?          # Required for Tier 2 and 3
|-- REMARK.md?             # Required for Tier 2 and 3
`-- binder/
    |-- environment.yml
#   |-- {optional binder files}?

# ? indicates tier-specific or optional requirements
```

## Tier-Specific Requirements

### Tier 1: Docker REMARK

**Purpose**: Minimal reproducibility via Docker containerization.

**Requirements:**
- All base requirements listed above
- **`README.md`**: Must be at least 50 lines (non-empty lines)
- **`CITATION.cff`**: Optional but recommended

**Use Case**: Suitable for exploratory work or early-stage projects that need basic containerization.

---

### Tier 2: Reproducible REMARK

**Purpose**: Enhanced reproducibility with comprehensive documentation and metadata.

**Requirements:**
- All Tier 1 requirements
- **`README.md`**: Must be at least 100 lines (non-empty lines) with comprehensive documentation
- **`REMARK.md`**: Required metadata file that should specify the REMARK tier and include additional project metadata
- **`CITATION.cff`**: Required. A valid [CITATION.cff](https://citation-file-format.github.io/) document with bibliographic metadata for the repository. To create a CITATION.cff file, you may use the [citation file format initializer](https://citation-file-format.github.io/cff-initializer-javascript/#/start)

**Use Case**: Suitable for research projects that are ready for sharing and replication but not yet published.

---

### Tier 3: Published REMARK

**Purpose**: Publication-ready with permanent archival and citation support.

**Requirements:**
- All Tier 2 requirements
- **`README.md`**: Must be at least 100 lines (non-empty lines) with comprehensive documentation
- **`REMARK.md`**: Must specify `tier: 3` in the metadata
- **`CITATION.cff`**: Must include citation information. DOI is recommended but not strictly required for Tier 3 LCD (Lowest Common Denominator) certification.
- **Zenodo DOI**: Obtain a permanent Zenodo DOI for your repository
- **Specific Git Tag**: The Zenodo archive must correspond to a specific git tag (e.g., `v1.0.0`)
- **Version Verification**: The tag provides cryptographic proof that the econ-ark fork matches your Zenodo archive exactly

**📋 Complete Step-by-Step Guide**: See [ZENODO-GUIDE.md](ZENODO-GUIDE.md) for detailed instructions with troubleshooting

**Quick Process Summary**:
1. Create a git tag for the version to archive: `git tag -a v1.0.0 -m "Published version"`
2. Push the tag to GitHub: `git push origin v1.0.0`
3. Enable Zenodo-GitHub integration (see [ZENODO-GUIDE.md](ZENODO-GUIDE.md) for detailed setup)
4. Create a GitHub release from that tag, which triggers Zenodo archival
5. Obtain the Zenodo DOI from the archived version
6. Add the DOI to your CITATION.cff: `doi: 10.5281/zenodo.XXXXXX`
7. When submitting to econ-ark/REMARK, specify the exact git tag that was archived

**Why This Matters**:
- The git tag creates an immutable snapshot at a specific commit
- The commit SHA-1 hash provides cryptographic verification
- Anyone can verify versions match by comparing commit hashes
- This guarantees: econ-ark fork = Zenodo archive = your tagged version

**Timeline**: 30-45 minutes following the [ZENODO-GUIDE.md](ZENODO-GUIDE.md) checklist

**Use Case**: Suitable for research that is being submitted for publication or has been published, requiring permanent archival and citation support.

## Submitting a REMARK

To index your repository as a REMARK, please file a Pull Request in [this repository](https://github.com/econ-ark/REMARK).
Your Pull Request should either create a new catalog entry or increment the existing version entry for your REMARK.

Once you open a Pull Request, the Econ-ARK team will review your submission to ensure that your `reproduce.sh` script
is able to run in the provided environment. If you would like to test this process on your own, you may clone this
repository, update the cataloged version, and build/execute your REMARK using the provided `cli.py` script.

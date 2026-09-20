# AI Context: REMARK Repository

## Repository Overview

This repository defines and maintains the **REMARK standard** -- a specification
for creating reproducible computational economics research. REMARK stands for
"R[eplications/eproductions] and Explorations Made using ARK".

## Core Purpose

- **Primary Function**: Standardize reproducible computational economics research
- **Problem Solved**: Eliminates the common frustration of downloading economics
  papers with code that cannot be reproduced on different systems
- **Target Audience**: Economics researchers, computational economists, academics
  working with the Econ-ARK toolkit

## Repository Structure & Key Components

### Documentation Files

- `README.md`: Main project overview and catalog information
- `STANDARD.md`: Technical requirements for REMARK compliance -- the canonical
  source of truth for tier definitions and required files
- `WORKFLOW.md`: How the catalog entries flow to the econ-ark.org website
- `Motivation.md`: Why the standard was created
- `How-To-Make-A-REMARK.md`: Technical instructions for creating REMARKs
- `EDITORIAL.md`: Guidelines for editors reviewing REMARK submissions
- `FAQ.md`: Common questions about the project

### Technical Infrastructure

- `cli.py`: Command-line tool for cloning, building, linting, and executing REMARKs
- `myst.yml`: MyST configuration for documentation site generation
- `requirements.txt`: Python dependencies for the CLI tool
- `REMARKs/`: Directory containing YAML metadata files (`.yml`) for each cataloged REMARK

### Content Types

The repository catalogs three types of computational economics projects:

1. **Explorations**: Demonstrations using Econ-ARK/HARK toolkit
2. **Replications**: Attempts to replicate published papers or canonical results
3. **Reproductions**: Code reproducing ALL results of a paper

## Compliance Tiers (from STANDARD.md)

REMARKs are organized into three tiers. All tiers require: tagged release,
`Dockerfile`, `reproduce.sh`, `README.md`, `LICENSE`, `binder/environment.yml`,
and a committed lockfile with pinned dependency versions (e.g. `uv.lock`,
`poetry.lock`, compiled `requirements.txt`, fully pinned `environment.yml`,
or `conda-lock.yml`). `uv` is recommended for new projects; see `STANDARD.md`.

| Tier | Name | Additional Requirements |
|------|------|------------------------|
| 1 | Docker REMARK | README >= 50 lines; `CITATION.cff` optional |
| 2 | Reproducible REMARK | README >= 100 lines; `CITATION.cff` + `REMARK.md` required |
| 3 | Published REMARK | Tier 2 + Zenodo DOI + specific git tag |

## Key Differentiators

- **vs DemARK**: REMARKs can rely on local files/subdirectories with predictable filepaths
- **vs Traditional Papers**: Requires computational reproducibility as a core standard

## Automation & Tools

- CLI tool supports parallel processing with `-J` flag
- Automated environment building (conda/docker)
- Automated linting to check standard compliance
- Integration with MyBinder for cloud execution
- MyST-based documentation site generation

## Governance & Quality Control

- Pull request-based submission process: authors add a catalog entry
  (`REMARKs/{name}.yml`) pointing to their repo via PR; on acceptance,
  Econ-ARK will fork the author's repo and the catalog/website point to that
  fork until the author submits a new version; Econ-ARK will then update the
  fork as long as `reproduce.sh` runs and the draft still meets REMARK requirements.
  Authors keep ownership.
- Editorial review for standard compliance
- Public catalog at econ-ark.org/materials
- Minimal review for successful replications
- Additional scrutiny for replications with discrepancies

## Data Format

REMARK metadata stored as YAML files with fields:

- `name`: Short identifier
- `remote`: GitHub repository URL
- `title`: Human-readable title
- Additional fields in corresponding markdown files for website display

## Integration Points

- **Econ-ARK website**: Displays catalog at econ-ark.org/materials
- **MyBinder**: Enables cloud execution of REMARKs
- **Conda/Docker**: Supports multiple environment types
- **GitHub**: Uses releases, pull requests, and standard Git workflows

## Success Metrics & Impact

- Addresses reproducibility difficulties in computational economics
- Enables "standing on shoulders of giants" rather than reinventing code
- Promotes transparency and collaboration in economics research
- Reduces time from weeks/months to "touch of a button" for running others' code

## Current Status

- Active repository with 20+ cataloged REMARKs
- Established workflow and tooling
- Integration with broader Econ-ARK ecosystem
- Ongoing submissions and maintenance

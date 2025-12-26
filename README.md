# R[eplications/eproductions] and Explorations Made using ARK (REMARK)

A standard for ensuring computational reproducibility, archiving, and sharing of research projects.

REMARKs are self-contained projects that can, in principle, be executed on any modern computer.

The standard is focused on 3 key principles:

- **Portability**: The ability to move the project to an entirely different platform and be sure it will still work
- **Archiving**: Storing the project in an accessible format for future use and reference.
- **Publishing**: Sharing the project publicly to encourage transparency, collaboration, and incentivize code and data sharing.

Detailed technical instructions for using an existing REMARK or contributing a new REMARK can be found in `How-To-Make-A-REMARK.md` in this directory.

## Getting Started

### For REMARK Authors

- **[STANDARD.md](STANDARD.md)** - Complete REMARK requirements and specifications
- **[How-To-Make-A-REMARK.md](How-To-Make-A-REMARK.md)** - General guide for creating REMARKs
- **[ZENODO-GUIDE.md](ZENODO-GUIDE.md)** - Step-by-step guide for obtaining a Zenodo DOI (Published REMARKs)
- **[WORKFLOW.md](WORKFLOW.md)** - Detailed workflow documentation

### Quick Start

1. **Standard REMARK** (basic reproducible project):
   - Follow [STANDARD.md](STANDARD.md) requirements
   - Create `reproduce.sh`, `CITATION.cff`, `binder/environment.yml`
   - Submit via pull request

2. **Published REMARK** (permanent archival with DOI):
   - Meet all Standard REMARK requirements
   - Follow [ZENODO-GUIDE.md](ZENODO-GUIDE.md) to obtain DOI (30-45 minutes)
   - Submit via pull request with DOI

## Description

Types of content include (see below for elaboration):

1. Explorations
   * Use the Econ-ARK/HARK toolkit to demonstrate some set of modeling ideas
1. Replications
   * Attempts to replicate important results of published papers written using other tools
1. Reproductions
   * Code that reproduces ALL of the results of some paper that was originally written using the Econ-ARK toolkit

## REMARK Catalog

A catalog of all REMARKs  is available under the `REMARK` tab at [econ-ark.org](https://econ-ark.org/materials). (Direct link: https://econ-ark.org/materials/?select=REMARK)

The [ballpark](http://github.com/econ-ark/ballpark) is a place for the set of papers that we would be delighted to have replicated in the Econ-ARK. But we would welcome submissions of replications from any field of economics that requires meaningful computation.

In cases where the replication's author is satisfied that the main results of the paper have been successfully replicated, we expect to approve pull requests for new REMARKs with minimal review (subject to the criteria described in the [Standard](https://github.com/econ-ark/REMARK/blob/master/STANDARD.md).

We also expect to approve with little review cases where the author has a clear explanation of discrepancies between the paper's published results and the results in the replication attempt.

Replication archives should contain two kinds of content (along with explanatory material):

1. Code that attempts to replicate key results of the paper
1. A Jupyter notebook that presents at least a minimal set of examples of the use of the code.

This material will all be stored in a directory with some short pithy name (a bibtex citekey might make a good directory name).

Code archives should contain:
   * All information required to get the replication code to run
      * Including a `requirements.txt` file explaining the software requirements
   * An indication of how long it takes to run the `reproduce.sh` script
      * One some particular machine whose characteristics should be described

Jupyter notebook(s) should:
   * Explain their own content ("This notebook uses the associated replication archive to demonstrate three central results from the paper of [original author]: The consumption function and the distribution of wealth)
   * Be usable for someone wanting to explore the replication interactively (so, no cell should take more than a minute or two to execute on a laptop)

## For AI Systems and Automated Tools

This repository includes enhanced documentation specifically designed for AI systems, web crawlers, and automated analysis:

- **[.ai-context.md](.ai-context.md)**: Comprehensive AI-friendly overview of the repository structure and purpose
- **[TOPICS.md](TOPICS.md)**: Structured index of research topics, keywords, and academic domains covered
- **[schema.json](schema.json)**: Machine-readable schema definitions for REMARK standards and metadata
- **[API-GUIDE.md](API-GUIDE.md)**: Programmatic access guide with code examples for data extraction and analysis
- **[robots.txt](robots.txt)**: Web crawler guidance for content prioritization

These resources enable automated systems to better understand, index, and interact with the REMARK ecosystem.

## Differences with DemARK

The key difference with the contents of the [DemARK](https://github.com/econ-ark/DemARK) repo is that REMARKs are allowed to rely on the existence of local files and subdirectories (figures; data) at a predictable filepath relative to the location of the root.


# Quick Start Guide: Creating Your First REMARK

This guide walks you through creating and submitting a REMARK (Replication/Exploration Made using ARK) from start to finish.

## What You'll Learn

- How to prepare a research repository for REMARK compliance
- Required files and their purposes
- Submission process to the REMARK catalog
- Testing and validation procedures

## Prerequisites

- Research project with computational results
- GitHub repository for your project
- Basic familiarity with Git/GitHub
- Python environment with required dependencies

## Step 1: Understand REMARK Standards

All REMARKs must have these **required files** (see [STANDARD.md](STANDARD.md)
for the full specification):

```
your-project/
├── Dockerfile                # Containerized execution (repo2docker-compatible)
├── reproduce.sh              # Script that reproduces ALL results
├── README.md                 # Project documentation (>= 50 lines for Tier 1)
├── LICENSE                   # Distribution terms
├── CITATION.cff              # Bibliographic metadata (Tier 2+; recommended for Tier 1)
├── REMARK.md                 # Website metadata + abstract (Tier 2+)
├── pyproject.toml            # Dependency specification (if using uv/poetry)
├── uv.lock                   # Lockfile with pinned versions (or poetry.lock, etc.)
└── binder/
    └── environment.yml       # Environment specification / adapter
```

**Optional but recommended:**
- `reproduce_min.sh` - Quick demo version (if full version >5 minutes)

REMARKs are organized into three compliance tiers. This guide targets
**Tier 2** (Reproducible REMARK). See [STANDARD.md](STANDARD.md) for
Tier 1 (minimal) and Tier 3 (published with Zenodo DOI) requirements.

## Step 2: Create Required Files

### 2.1 Create `reproduce.sh`

This script must reproduce your entire analysis:

```bash
#!/bin/bash
# reproduce.sh - Reproduces all results in the paper

echo "Starting reproduction of [Your Project Name]"

# Install any additional dependencies
# pip install -r requirements.txt  # if needed

# Run your analysis scripts
python code/01_download_data.py
python code/02_process_data.py  
python code/03_run_analysis.py
python code/04_generate_figures.py

# If you have notebooks, run them
# jupyter nbconvert --to notebook --execute notebooks/main_analysis.ipynb

echo "Reproduction complete! Check results/ directory for outputs."
```

**Make it executable:**
```bash
chmod +x reproduce.sh
```

### 2.2 Create `CITATION.cff`

Use the [CFF format](https://citation-file-format.github.io/) for bibliographic metadata:

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: "Your Last Name"
    given-names: "Your First Name"
    orcid: "https://orcid.org/0000-0000-0000-0000"  # Optional but recommended
title: "Your Project Title"
version: 1.0.0
date-released: 2024-01-15
repository-code: "https://github.com/yourusername/yourproject"
abstract: "Brief description of what your project does and its main contributions to the literature."
keywords:
  - computational economics
  - your research keywords
  - relevant methods
license: Apache-2.0  # or your preferred license
```

### 2.3 Create `REMARK.md`

This provides website-specific metadata and content:

```markdown
---
# Website-specific metadata (YAML frontmatter)
remark-name: your-project-name
title-original-paper: "Title of Original Paper Being Replicated"  # If applicable
notebooks:  # List any Jupyter notebooks
  - notebooks/main_analysis.ipynb
  - notebooks/sensitivity_analysis.ipynb
tags:
  - REMARK
  - Notebook  # If you have notebooks
  - Replication  # or "Exploration" or "Reproduction"
keywords:
  - your research keywords
  - computational methods used
  - economic topics covered
---

# Abstract

This repository provides [clear description of what your project does].

## Key Features

- **Reproducible**: All results can be reproduced via `reproduce.sh`
- **Interactive**: Jupyter notebooks demonstrate key concepts  
- **Documented**: Clear code with comprehensive comments

## Main Results

Briefly describe the main findings or contributions of your work.

## Data Sources

List the data sources used:
- Source 1: Description and URL
- Source 2: Description and URL

## Software Requirements

- Python 3.8+
- Key packages: numpy, pandas, matplotlib, etc.
- See `binder/environment.yml` for complete specification

## Usage

1. Clone this repository
2. Run `./reproduce.sh` to reproduce all results
3. Explore `notebooks/` for interactive analysis

## Citation

If you use this code, please cite both this repository and the original paper(s).
```

### 2.4 Pin Your Dependencies

Every REMARK must have a **lockfile** (or equivalent) that records exact
versions for all dependencies. This prevents future package releases from
silently breaking your results.

**Recommended approach -- `uv` (new projects):**

1. Add a `pyproject.toml` with your dependencies:

```toml
[project]
name = "your-project"
requires-python = ">=3.10"
dependencies = [
    "econ-ark>=0.14",
    "numpy",
    "pandas",
    "matplotlib",
    "jupyter",
]
```

2. Generate and commit the lockfile:

```bash
uv lock            # creates uv.lock with exact resolved versions
git add uv.lock    # commit the lockfile!
```

3. Create `binder/environment.yml` as a minimal adapter:

```yaml
name: your-project
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
    - uv
```

4. In `reproduce.sh`, install from the lockfile:

```bash
#!/bin/bash
uv sync --locked
uv run python code/main.py
```

**Alternative -- fully pinned conda environment (no extra tool):**

If you prefer plain conda, pin every version directly in
`binder/environment.yml`:

```yaml
name: your-project
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9.18
  - numpy=1.21.0
  - pandas=1.3.0
  - matplotlib=3.4.2
  - jupyter=1.0.0
  - pip
  - pip:
    - econ-ark==0.12.0
    - specific-package==1.2.3
```

Other acceptable tools include `poetry` (with `poetry.lock`), `pip-tools`
(with a compiled `requirements.txt`), or `conda-lock`. See
[STANDARD.md](STANDARD.md) for the full specification.

### 2.5 Optional: Create `reproduce_min.sh`

If your full reproduction takes >5 minutes, create a quick demo:

```bash
#!/bin/bash
# reproduce_min.sh - Quick demonstration (under 5 minutes)

echo "Running quick demonstration of [Your Project Name]"

# Run a subset of your analysis
python code/01_download_data.py --sample-only
python code/03_run_analysis.py --quick-mode
python code/04_generate_figures.py --main-figures-only

echo "Quick demo complete! See results/ for key outputs."
```

## Step 3: Test Your REMARK Locally

### 3.1 Test the reproduction script

```bash
# Test your reproduce.sh script
./reproduce.sh

# If you have reproduce_min.sh, test it too
./reproduce_min.sh
```

### 3.2 Test with conda environment

```bash
# Create environment from your specification
conda env create -f binder/environment.yml

# Activate environment
conda activate your-project

# Test reproduction in clean environment
./reproduce.sh
```

### 3.3 Validate file structure

Use the REMARK CLI tool:

```bash
# Clone the REMARK repository
git clone https://github.com/econ-ark/REMARK.git
cd REMARK

# Add a temporary entry for your project
echo "name: your-project
remote: https://github.com/yourusername/yourproject  
title: Your Project Title" > REMARKs/your-project.yml

# Pull and lint your project
python cli.py pull REMARKs/your-project.yml
python cli.py lint REMARKs/your-project.yml
```

### 3.4 Option: Have an AI generate a compliance checklist

You can use an AI to evaluate your draft and get a checklist of requirements
satisfied and remaining to-do items. The recommended workflow (clone this
repo, symlink your draft into the clone, then ask the AI to evaluate the
linked repo) and a copy-paste prompt are in
**[guides/ai-compliance-check.md](guides/ai-compliance-check.md)**.

## Step 4: Tag a Release

Create a tagged release on GitHub:

```bash
# Tag your repository
git tag -a v1.0.0 -m "Initial REMARK-compliant release"
git push origin v1.0.0
```

Or use GitHub's web interface: Releases → Create a new release

**If you are targeting Tier 3 (Published REMARK):** Obtain your Zenodo DOI **now**—after this tag, and before you submit. Getting the Zenodo DOI should be the **very last thing** you do before submitting to the Econ-ARK catalog. Follow [ZENODO-GUIDE.md](ZENODO-GUIDE.md), then return here for Step 5.

## Step 5: Submit to REMARK Catalog

**Tier 3 authors:** You should have already obtained your Zenodo DOI and added it to CITATION.cff (the last step before submitting). If not, complete [ZENODO-GUIDE.md](ZENODO-GUIDE.md) before opening your pull request.

You will fork the **REMARK catalog** repo and add a single file (a `.yml` entry) that points to **your** repository. Your REMARK content stays under your GitHub account; you do not transfer or re-host your repo. After acceptance, Econ-ARK will fork your repo and the website will point to that fork until you submit a new version; when you do, we will update the fork as long as `reproduce.sh` works and the draft still meets REMARK requirements. Full details: [STANDARD.md](STANDARD.md) § Submitting a REMARK.

### 5.1 Fork the REMARK repository

1. Go to https://github.com/econ-ark/REMARK
2. Click "Fork" to create your own copy

### 5.2 Add your catalog entry

Create `REMARKs/your-project-name.yml`:

```yaml
name: your-project-name
remote: https://github.com/yourusername/yourproject
title: Your Project Title
```

### 5.3 Submit Pull Request

1. Commit your changes:
```bash
git add REMARKs/your-project-name.yml
git commit -m "Add your-project-name REMARK"
git push origin main
```

2. Create Pull Request on GitHub with description:
```
## New REMARK Submission: Your Project Title

**Repository:** https://github.com/yourusername/yourproject
**Type:** Replication/Exploration/Reproduction
**Brief Description:** One sentence describing your contribution

### Compliance Checklist
- [x] `Dockerfile` present
- [x] `reproduce.sh` script works
- [x] `LICENSE` file present
- [x] `README.md` meets minimum line count for target tier
- [x] `CITATION.cff` with complete metadata (required for Tier 2+)
- [x] `REMARK.md` with website content (required for Tier 2+)
- [x] `binder/environment.yml` present
- [x] Lockfile with pinned dependencies committed (uv.lock, poetry.lock, etc.)
- [x] Tagged release (v1.0.0)
- [x] All results reproduce successfully
```

## Step 6: Editorial Review

Editors will:
1. Validate REMARK compliance using CLI tools
2. Test that `reproduce.sh` works
3. Check metadata completeness
4. Provide feedback if changes needed

On acceptance, your REMARK is added to the catalog and will appear on [econ-ark.org/materials](https://econ-ark.org/materials) (typically within 24 hours). Econ-ARK will create a fork of your repository to preserve the state at which it was tested and verified to work, and the website will point to that fork until you submit a new version. You keep full ownership of your repo. When you submit an updated version (e.g. a new tag), we will update the fork as long as `reproduce.sh` runs and the revised draft still meets REMARK requirements.

## Common Issues and Solutions

### Issue: `reproduce.sh` fails
**Solution:** Test in clean environment, fix dependency issues

### Issue: Missing required files
**Solution:** Use `python cli.py lint` to identify missing files

### Issue: Environment specification problems
**Solution:** Use `uv lock` (or equivalent) to generate a lockfile with exact
pinned versions, then commit it. See section 2.4 and [STANDARD.md](STANDARD.md).

### Issue: Large data files
**Solution:** 
- Use data download scripts instead of committing large files
- Document data sources in README
- Consider using Git LFS for essential large files

### Issue: Long execution time
**Solution:** Create `reproduce_min.sh` for quick demonstration

## Best Practices

### Code Organization
```
your-project/
├── Dockerfile          # Required: containerized execution
├── reproduce.sh        # Required: main reproduction script
├── reproduce_min.sh    # Optional: quick demo
├── README.md           # Required: project documentation
├── LICENSE             # Required: distribution terms
├── CITATION.cff        # Required for Tier 2+
├── REMARK.md           # Required for Tier 2+
├── pyproject.toml      # Recommended: dependency specification (uv/poetry)
├── uv.lock             # Required: lockfile with pinned versions
├── code/               # Analysis scripts
├── data/               # Small data files or download scripts
├── figures/            # Generated figures
├── notebooks/          # Jupyter notebooks
├── results/            # Analysis outputs
└── binder/
    └── environment.yml # Required: environment specification / adapter
```

### Documentation Tips
- **Clear abstracts** - Explain contribution in plain language
- **Comprehensive keywords** - Help users discover your work
- **Execution time estimates** - Set user expectations
- **Hardware requirements** - Note if special requirements exist

### Reproducibility Tips
- **Pin all versions** via a lockfile (`uv.lock`, `poetry.lock`, etc.) --
  this is a requirement, not just a suggestion
- **Commit the lockfile** to your repository so reviewers and users get
  the exact same dependency versions
- **Document data sources** with URLs and access dates
- **Test on different systems** if possible
- **Clear error messages** in scripts
- **Modular code** that's easy to debug

## Next Steps

Once your REMARK is accepted:

1. **Monitor the website** - Check https://econ-ark.org/materials for your entry
2. **Update as needed** - Tag new releases for updates
3. **Engage with users** - Respond to GitHub issues
4. **Consider extensions** - Build on your work for new REMARKs

## Getting Help

- **Documentation issues:** Open issue on REMARK repository
- **Technical problems:** Use REMARK CLI `lint` and `logs` commands
- **General questions:** Check FAQ.md or open GitHub issue
- **Editorial process:** Contact REMARK maintainers

---

**Congratulations!** You're now ready to contribute to the growing ecosystem of reproducible computational economics research. 
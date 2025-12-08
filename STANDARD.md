# REMARK Guidelines

A REMARK refers to an open git repository that is indexed in this repository with appropriate metadata.

---

## Three-Tier REMARK System

REMARKs are organized into **three progressive tiers** that build on Docker-based reproducibility:

### **Tier 1: Docker REMARK**
- **Focus**: Minimal requirements for Docker-based reproduction
- **Philosophy**: "If you can run Docker, you can reproduce the computational results"
- **Typical Use**: Work-in-progress, graduate student projects, exploratory research

### **Tier 2: Reproducible REMARK**
- **Focus**: Enhanced inspectability and experimentation
- **Philosophy**: "You can understand my code, modify parameters, and experiment"
- **Typical Use**: Working papers, dissertation chapters, mature research projects

### **Tier 3: Published REMARK**
- **Focus**: Ready for submission to ANY top economics journal (LCD requirements)
- **Philosophy**: "Meets minimum requirements for all major journals"
- **Typical Use**: Papers ready for journal submission, final dissertations

**Key principle**: Each tier builds on the previous. All tiers prioritize computational reproducibility; **none require LaTeX document reproduction**.

---

## Tier 1: Docker REMARK

### Minimal Repository Structure

```bash
.
|-- Dockerfile
|-- reproduce.sh
|-- README.md         # ≥50 lines
|-- LICENSE
`-- binder/
    `-- environment.yml
```

### Tier 1 Requirements

1. **Dockerfile** that builds computational environment successfully
2. **Docker build instructions** in README (copy-paste ready)
3. **Docker run instructions** in README (copy-paste ready)
4. **README.md** (≥50 lines) with:
   - Project title and brief description
   - Docker build command
   - Docker run command
   - Expected outputs (brief)
   - System requirements (Docker version)
5. **LICENSE** file with open source license
6. **reproduce.sh** script that runs computational analysis in Docker
7. **Verified** by author that Docker build and run work successfully

**Not Required**:
- REMARK.md metadata
- CITATION.cff
- LaTeX document reproduction
- Enhanced documentation sections
- Specific data formats
- Academic metadata

---

## Tier 2: Reproducible REMARK

### Enhanced Repository Structure

```bash
.
|-- Dockerfile
|-- reproduce.sh
|-- reproduce_min.sh?
|-- README.md         # ≥100 lines with required sections
|-- REMARK.md         # Metadata for discovery
|-- CITATION.cff      # Citation information
|-- LICENSE
`-- binder/
    `-- environment.yml
```

### Tier 2 Requirements

**All Tier 1 requirements PLUS**:

1. **Enhanced README** (≥100 lines) with sections:
   - **Installation Instructions**: Environment setup (Docker and/or native)
   - **Reproduction Instructions**: Step-by-step reproduction guide
   - **Code Organization**: Directory structure and key files
   - **Parameter Modification Guide**: How to modify parameters and run variants
   - **Output Guide**: What outputs are generated and where to find them

2. **REMARK.md** with metadata:
   ```yaml
   ---
   github_repo_url: [REPO_URL]
   remark-name: [PROJECT_NAME]
   tier: 2  # Specify tier
   notebooks:  # if applicable
     - notebook.ipynb
   tags:
     - REMARK
     - Reproducible
   keywords:
     - [3-5 keywords]
   ---
   Brief description
   ```

3. **CITATION.cff** with:
   - title
   - authors (names, affiliations)
   - repository-code URL
   - keywords

4. **Plain-text data access**:
   - CSV/TXT/JSON/TSV for data, OR
   - Conversion scripts to generate plain-text from proprietary formats

5. **Commented code**: Functions and non-obvious logic documented

6. **Organized code structure**: Logical file organization, meaningful names

**Not Required** (Tier 3+ Enhanced only):
- Zenodo DOI (recommended for Tier 3, required for Enhanced)
- Strict 7-section README structure (content matters more than format)
- CSV-only data (any accessible format OK)
- Exact patch-level version pinning (major versions sufficient)
- Multiple platform testing (one platform sufficient for Tier 3)
- Output mapping table (clear documentation sufficient)

---

## Tier 3: Published REMARK (LCD Definition)

### Journal-Ready Repository Structure

```bash
.
|-- Dockerfile
|-- reproduce.sh
|-- reproduce_min.sh?
|-- README.md         # Comprehensive (≥100 lines recommended)
|-- REMARK.md         # Metadata with tier: 3
|-- CITATION.cff      # Citation information
|-- LICENSE
`-- binder/
    `-- environment.yml
```

### Core Principle

**"If it meets Tier 3, it will satisfy the MINIMUM requirements for submission to ANY top economics journal"**

Tier 3 represents the **least common denominator (LCD)** across:
- American Economic Review (AER)
- Quantitative Economics (QE)
- Econometrica (ECMA)
- Review of Economic Studies (REStud)
- Journal of Political Economy (JPE)
- Quarterly Journal of Economics (QJE)
- Review of Economics and Statistics (REStat)

### Tier 3 Requirements

**All Tier 2 requirements PLUS**:

#### 1. **Comprehensive README**
README with all essential reproduction information:
- Project title and authors
- Software requirements with major versions (e.g., Python 3.9)
- Installation instructions
- Reproduction instructions (step-by-step how to run)
- Expected runtime estimates
- Data access information
- Citation information

**Format**: Markdown acceptable  
**Structure**: Flexible - content matters more than specific sections  
**Length**: ≥100 lines typical for comprehensive coverage  
**Acceptable**: Can link to separate documents (INSTALLATION.md, etc.)

#### 2. **Complete Software Environment Specification**
Document all software needed to reproduce results:
- Programming language with major version (Python 3.9, R 4.1)
- Critical research packages with exact versions (econ-ark==0.14.1)
- Supporting packages can use ranges (numpy>=1.24,<2)
- Operating system tested on

**Formats**: environment.yml, requirements.txt, pyproject.toml, Dockerfile, or README  
**NOT Required**: Exact patch versions for all packages, strict lockfiles

#### 3. **Data Availability Compliance**
Data included OR complete access instructions:

**If Data Included**:
- Any commonly accessible format: CSV, .dta (Stata), .xlsx (Excel), etc.
- Variable documentation (codebook, labels, or README)
- Data citations

**If Data Restricted**:
- Clear access instructions (where, how, cost)
- Test/synthetic data for code verification
- Documentation of restrictions

**NOT Required**: CSV-only format (proprietary formats acceptable if common)

#### 4. **Tested Reproduction Verification**
Author has verified reproduction works:
- Code runs without errors
- Produces expected outputs
- Tested on at least one platform
- README states testing was performed
- Known issues documented

**Acceptable**: Single platform testing (multiple recommended but not required)

#### 5. **Complete Analysis Code**
All code to reproduce results included:
- Data cleaning/processing code
- Main analysis code
- Figure/table generation code
- Helper functions and utilities
- Comments explaining non-obvious logic
- Clear entry points (main.py, reproduce.sh)

**NOT Required**: Specific directory structure, unit tests, type annotations

#### 6. **Accessible Public Repository**
Code available on publicly accessible platform:
- GitHub/GitLab (public repository)
- Zenodo, figshare, OSF
- Journal-hosted repositories
- Institutional repositories

**NOT Required**: Specific platform, DOI (though recommended)

#### 7. **Replication-Permissive License**
LICENSE file permitting research use:
- Apache 2.0, MIT, BSD, CC-BY, or other OSI-approved
- Must permit running code for verification
- Must permit academic replication
- Must permit modification for research

#### 8. **Citation Information**
Clear citation instructions in any standard format:
- CITATION.cff file (recommended)
- Citation block in README
- CITATION.txt or CITATION.md
- BibTeX entry in README

**Must Include**: Authors, title, year, repository URL  
**Optional** (add after publication): Paper DOI, journal info

---

### What's Recommended (Not Required)

These are **best practices** but NOT minimum LCD requirements:

- **Zenodo DOI**: Excellent for permanence (journals use different platforms)
- **Exact version pinning**: Good for long-term reproducibility (ranges acceptable)
- **CSV-only data**: Best for longevity (.dta, .xlsx acceptable)
- **Specific README structure**: Helpful (content matters more than format)
- **Multiple platform testing**: Ideal (one platform sufficient)
- **Output mapping table**: Clear (explicit mapping not universal)

---

### For Stricter Journals (Tier 3+ Enhanced)

Some journals have MORE stringent requirements. See separate guide:
**"Tier-3-Enhanced-Guide.md"** for upgrades needed for:
- Quantitative Economics (QE): Requires Zenodo DOI, 7-section README, CSV data
- American Economic Review (AER): Requires openICPSR archival
- Journal-specific templates and formats

---

## General Requirements (All Tiers)

### Tagged Releases

Have a [tagged release](https://docs.github.com/en/github/administering-a-repository/managing-releases-in-a-repository).
- If you make changes to your work, please release a new version/tag.
- Tags commonly follow [semantic versioning](https://en.wikipedia.org/wiki/Software_versioning#Semantic_versioning): `v{major}.{minor}.{patch}`

### Binder Configuration

A `binder/` directory containing configuration files to recreate your execution environment:
- **`binder/environment.yml`** file for non-interactive environment creation
  - May be a fully specified conda environment with pinned dependencies, OR
  - A minimal conda environment that installs your environment manager (`uv`, `poetry`, `pip-tools`) and any needed tools
- Optional: Other [binder configuration files](https://mybinder.readthedocs.io/en/latest/using/config_files.html)

### Reproduction Script

A `reproduce.sh` script that:
- Runs and reproduces all computational results
- Returns exit code 0 on success
- Works inside Docker container

### Optional: Minimal Reproduction

`reproduce_min.sh` (optional):
- Include if your full `reproduce.sh` takes ≥5 minutes
- Provides quick verification that code runs

---

## Submitting a REMARK

### Tier Selection

When submitting, specify your tier in the Pull Request:
- **Tier 1**: For rapid sharing with minimal documentation
- **Tier 2**: For mature projects with good documentation
- **Tier 3**: For published work requiring permanent citation

### Submission Process

1. **Prepare your repository** according to your chosen tier's requirements
2. **Test locally** using the REMARK CLI tool:
   ```bash
   remark lint /path/to/your/repo --tier=1  # or --tier=2, --tier=3
   ```
3. **File a Pull Request** in [this repository](https://github.com/econ-ark/REMARK)
   - Create or update catalog entry in `REMARKs/[YourProject].yml`
   - Specify tier in metadata
4. **Review**: Econ-ARK team will verify your `reproduce.sh` runs successfully

### Upgrading Tiers

You can upgrade your REMARK to a higher tier at any time:
- Submit a new PR with updated tier specification
- Ensure all requirements for the new tier are met

---

## Testing Your REMARK

Clone this repository and test your REMARK:

```bash
# Clone REMARK repository
git clone https://github.com/econ-ark/REMARK
cd REMARK

# Test your REMARK at specific tier
./cli.py lint /path/to/your/repo --tier=1

# Build and execute (all tiers)
./cli.py build docker /path/to/your/repo
./cli.py execute docker /path/to/your/repo
```

---

## Tier Comparison Summary

| Requirement | Tier 1 | Tier 2 | Tier 3 (LCD) |
|-------------|--------|--------|--------------|
| Dockerfile | ✅ | ✅ | ✅ |
| reproduce.sh | ✅ | ✅ | ✅ |
| README length | ≥50 lines | ≥100 lines | ≥100 lines |
| Docker instructions | ✅ | ✅ | ✅ |
| LICENSE | ✅ | ✅ | ✅ |
| REMARK.md | ❌ | ✅ | ✅ |
| CITATION.cff | ❌ | ✅ | ✅ |
| Parameter Guide | ❌ | ✅ | ✅ |
| Data format | Any | CSV or accessible | Any accessible format |
| Code comments | Basic | ✅ | ✅ |
| Software versions | Any | Documented | Major + critical exact |
| Testing | Docker | Docker | ≥1 platform verified |
| Data access info | No | Helpful | Required in README |
| Citation info | No | Yes | Yes (flexible format) |
| **DOI** | **No** | **No** | **Recommended (not required)** |
| **Exact version pinning** | **No** | **No** | **Major + critical only** |
| **CSV-only data** | **No** | **No** | **No (.dta/.xlsx OK)** |

---

## Additional Resources

- **Three-Tier Proposal**: See `REMARK-THREE-TIER-PROPOSAL.md` for detailed rationale
- **How-To Guide**: See `How-To-Make-A-REMARK.md` for step-by-step instructions
- **CLI Guide**: See `CLI-GUIDE.md` for tool usage
- **Templates**: Coming soon - tier-specific README templates

---

## Questions?

- Open an issue: https://github.com/econ-ark/REMARK/issues
- Email: econ-ark@jhu.edu

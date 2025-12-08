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
- **Focus**: Publication-ready research archive with permanent DOI
- **Philosophy**: "You can cite my work permanently with confidence"
- **Typical Use**: Published papers, final dissertations, citable archives

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

**Not Required** (Tier 3 only):
- Zenodo DOI
- Data Availability Statement
- Formal Data Citations section
- Computational Requirements (exact versions)
- Output Mapping (code → specific tables/figures)
- Academic metadata (abstract, JEL codes)

---

## Tier 3: Published REMARK

### Publication-Ready Repository Structure

```bash
.
|-- Dockerfile
|-- reproduce.sh
|-- reproduce_min.sh?
|-- README.md         # ≥100 lines with 7 required sections
|-- REMARK.md         # Enhanced metadata with DOI
|-- CITATION.cff      # Enhanced with DOI and metadata
|-- LICENSE
`-- binder/
    `-- environment.yml
```

### Tier 3 Requirements

**All Tier 2 requirements PLUS**:

1. **Zenodo DOI**: Permanent archive on Zenodo
   - DOI in CITATION.cff
   - DOI badge in README
   - GitHub release created

2. **Complete README** with 7 required sections:
   - All Tier 2 sections PLUS:
   - **Data Availability Statement**: Access procedures, restrictions, costs
   - **Computational Requirements**: Exact versions, hardware, runtime, OS tested
   - **Output Mapping**: Which code generates which table/figure
   - **Data Citations**: Dedicated references section with bibliographic format

3. **Plain-text data** (strict):
   - CSV/TXT/JSON required for ALL shareable data
   - Proprietary formats may supplement but cannot replace
   - Variable documentation (codebooks) required

4. **Exact version specification**:
   - Exact Python/R version documented
   - Exact versions for critical research packages
   - README documents tested versions

5. **Code portability**:
   - No hardcoded absolute paths
   - Relative paths from repository root
   - Forward slashes for cross-platform compatibility
   - Tested on multiple platforms

6. **Data transformation documentation**:
   - All data cleaning/transformation code included
   - Clear workflow documentation
   - Separation from analysis code

7. **Academic metadata** in CITATION.cff or paper:
   - Abstract (≤150 words)
   - Keywords (3-8)
   - JEL codes (up to 3)

8. **Production-quality reproduction script**:
   - Orchestrates all steps
   - Generates log files
   - Cross-platform compatible

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

| Requirement | Tier 1 | Tier 2 | Tier 3 |
|-------------|--------|--------|--------|
| Dockerfile | ✅ | ✅ | ✅ |
| reproduce.sh | ✅ | ✅ | ✅ |
| README length | ≥50 lines | ≥100 lines | ≥100 lines |
| Docker instructions | ✅ | ✅ | ✅ |
| LICENSE | ✅ | ✅ | ✅ |
| REMARK.md | ❌ | ✅ | ✅ |
| CITATION.cff | ❌ | ✅ | ✅ Enhanced |
| Parameter Guide | ❌ | ✅ | ✅ |
| Plain-text data | ❌ | Scripts OK | CSV required |
| Code comments | Basic | ✅ | ✅ |
| Zenodo DOI | ❌ | ❌ | ✅ Required |
| Data Avail. Statement | ❌ | ❌ | ✅ |
| Exact versions | ❌ | ❌ | ✅ |
| Cross-platform | Docker only | Docker only | ✅ |
| Academic metadata | ❌ | ❌ | ✅ |

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

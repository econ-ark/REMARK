# [Project Title] - Reproducible REMARK (Tier 2)

**Brief Description**: [1-2 sentence description of your project]

**Authors**: [Author names and affiliations]

**Keywords**: [keyword1], [keyword2], [keyword3]

**Status**: Reproducible REMARK (Tier 2) - Enhanced inspectability and experimentation

---

## Quick Start

### Docker (Recommended)

```bash
# Build and run
docker build -t [project-name] .
docker run --rm -v $(pwd)/output:/app/output [project-name]
```

### Native Installation

```bash
conda env create -f binder/environment.yml
conda activate [env-name]
./reproduce.sh
```

---

## Table of Contents

1. [Overview](#overview)
2. [Installation Instructions](#installation-instructions)
3. [Reproduction Instructions](#reproduction-instructions)
4. [Code Organization](#code-organization)
5. [Parameter Modification Guide](#parameter-modification-guide)
6. [Output Guide](#output-guide)
7. [System Requirements](#system-requirements)
8. [Citation](#citation)

---

## Overview

### Research Question

[Describe the main research question or problem]

### Methods

[Describe the computational methods, models, or analyses used]

### Key Results

[Summarize main findings or outputs]

---

## Installation Instructions

### Option 1: Docker (Recommended)

1. **Install Docker**:
   - Download from https://www.docker.com/
   - Verify installation: `docker --version`

2. **Build Docker image**:
   ```bash
   docker build -t [project-name] .
   ```

### Option 2: Conda/Native

1. **Install Conda**:
   - Download from https://docs.conda.io/en/latest/miniconda.html

2. **Create environment**:
   ```bash
   conda env create -f binder/environment.yml
   ```

3. **Activate environment**:
   ```bash
   conda activate [env-name]
   ```

4. **Verify installation**:
   ```bash
   python --version
   python -c "import [key_package]; print([key_package].__version__)"
   ```

---

## Reproduction Instructions

### Full Reproduction

```bash
# In Docker
docker run --rm -v $(pwd)/output:/app/output [project-name]

# Native
./reproduce.sh
```

**Expected Runtime**: ~[X] hours

### Minimal Verification (Optional)

```bash
# In Docker
docker run --rm -v $(pwd)/output:/app/output [project-name] ./reproduce_min.sh

# Native
./reproduce_min.sh
```

**Expected Runtime**: ~[X] minutes

### Step-by-Step Manual Reproduction

If you prefer to run steps individually:

1. **Data preparation** (if needed):
   ```bash
   python code/prepare_data.py
   ```

2. **Main analysis**:
   ```bash
   python code/main_analysis.py
   ```

3. **Generate figures**:
   ```bash
   python code/create_figures.py
   ```

---

## Code Organization

### Directory Structure

```
.
├── code/
│   ├── prepare_data.py      # Data cleaning and preparation
│   ├── main_analysis.py     # Core computational analysis
│   ├── create_figures.py    # Visualization generation
│   └── utils.py             # Helper functions
├── data/
│   ├── raw/                 # Input data
│   └── processed/           # Cleaned data (generated)
├── output/
│   ├── figures/             # Generated figures
│   ├── tables/              # Generated tables
│   └── results/             # Numerical results
├── binder/
│   └── environment.yml      # Environment specification
├── Dockerfile               # Docker configuration
├── reproduce.sh             # Main reproduction script
├── reproduce_min.sh         # Quick verification script
├── REMARK.md               # REMARK metadata
├── CITATION.cff            # Citation information
└── README.md               # This file
```

### Key Files

- **`code/main_analysis.py`**: Entry point for analysis
  - Runs [description of what it does]
  - Calls helper functions from `utils.py`
  - Saves results to `output/results/`

- **`code/create_figures.py`**: Figure generation
  - Creates [number] figures
  - Uses results from main analysis
  - Saves to `output/figures/`

[Add descriptions for other key files]

---

## Parameter Modification Guide

### Where Parameters Are Defined

Key parameters are defined in:
- **`code/config.py`** or **`code/parameters.py`**
- Top of **`code/main_analysis.py`**

### Example: Modifying Key Parameters

#### 1. Change Discount Factor

In `code/parameters.py`:
```python
# Original
BETA = 0.96

# Modified (more patient agents)
BETA = 0.98
```

#### 2. Change Sample Size

In `code/main_analysis.py`:
```python
# Original
N_SIMULATIONS = 10000

# Modified (faster for testing)
N_SIMULATIONS = 1000
```

#### 3. Change Model Specification

In `code/config.py`:
```python
# Original
USE_STICKY_EXPECTATIONS = True

# Modified (turn off feature)
USE_STICKY_EXPECTATIONS = False
```

### Running Variants

After modifying parameters:

```bash
# Docker
docker build -t [project-name] .  # Rebuild with changes
docker run --rm -v $(pwd)/output:/app/output [project-name]

# Native
./reproduce.sh
```

### Common Modifications

1. **Robustness checks**: Try different parameter values
2. **Subsample analysis**: Reduce sample size for faster testing
3. **Alternative specifications**: Toggle model features on/off
4. **Sensitivity analysis**: Vary key parameters systematically

---

## Output Guide

### Generated Files

Running `reproduce.sh` generates:

#### Figures
- **`output/figures/figure1_[description].pdf`**: [Description]
- **`output/figures/figure2_[description].pdf`**: [Description]
- [etc.]

#### Tables
- **`output/tables/table1_[description].csv`**: [Description]
- **`output/tables/table2_[description].csv`**: [Description]
- [etc.]

#### Results
- **`output/results/main_results.json`**: [Description]
- **`output/results/statistics.csv`**: [Description]

### Interpreting Results

[Explain how to interpret key outputs, what values to expect, etc.]

---

## System Requirements

### Minimum Requirements

- **OS**: Linux, macOS, or Windows (with Docker or WSL2)
- **Docker**: Version 20.0+
- **RAM**: [X]GB
- **Disk**: [X]GB free space
- **Runtime**: ~[X] hours (full), ~[X] minutes (minimal)

### Tested Platforms

- ✅ macOS 12+ (M1/M2 and Intel)
- ✅ Ubuntu 20.04+
- ✅ Windows 10+ with WSL2

---

## Data

### Data Sources

[List data sources used, with access information]

### Data Format

- Input data: [CSV/etc.], located in `data/raw/`
- Processed data: Generated by `code/prepare_data.py`

---

## Troubleshooting

### Docker Issues

**Problem**: Docker build fails  
**Solution**: Ensure Docker daemon is running and you have sufficient disk space

**Problem**: Permission errors in output directory  
**Solution**: Check that output directory has write permissions

### Native Installation Issues

**Problem**: Environment creation fails  
**Solution**: Try creating environment with `conda env create -f binder/environment.yml --force`

**Problem**: Missing package errors  
**Solution**: Ensure you've activated the correct conda environment

---

## Citation

If you use this code or data, please cite:

```bibtex
@software{[citation_key],
  author = {[Author names]},
  title = {[Project title]},
  year = {[Year]},
  publisher = {GitHub},
  url = {[repository URL]}
}
```

Or use the `CITATION.cff` file for automatic citation generation.

---

## License

See [LICENSE](LICENSE) file for details.

---

## Contact

- **Issues**: [[repository URL]/issues]
- **Email**: [contact email]
- **Website**: [project website if available]

---

## Acknowledgments

[Acknowledge funding, data sources, collaborators, etc.]

---

**REMARK Tier**: 2 (Reproducible REMARK)  
**REMARK Name**: [project-name]  
**Last Updated**: [Date]  
**Repository**: [GitHub URL]

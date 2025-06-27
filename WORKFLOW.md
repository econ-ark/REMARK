# REMARK Ecosystem Workflow

This document provides a comprehensive overview of how the REMARK ecosystem works, including the interactions between the REMARK repository, individual research repositories, and the econ-ark.org website.

## System Architecture Overview

**⚠️ CRITICAL DISTINCTION**: The REMARK ecosystem has TWO SEPARATE SYSTEMS that serve different purposes:

1. **Website Generation System** (`populate_remarks.py`) - Generates econ-ark.org content
2. **REMARK Validation System** (`cli.py`) - Validates research reproducibility standards

**These are INDEPENDENT systems with different requirements!**

The REMARK ecosystem consists of three main components:

```monospace
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   REMARK Repo       │    │  Individual Repos   │    │  econ-ark.org       │
│  (Catalog/Standards) │    │  (Research Projects) │    │  (Public Website)   │
│                     │    │                     │    │                     │
│ • REMARKs/*.yml     │◄──►│ • CITATION.cff      │───►│ • _materials/*.md   │
│ • STANDARD.md       │    │ • REMARK.md         │    │ • Jekyll templates  │
│ • Validation tools  │    │ • reproduce.sh      │    │ • Search/filter UI  │
│ • CLI tools         │    │ • binder/env.yml    │    │ • Material pages    │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## Detailed Workflow

### 1. REMARK Repository Structure

The REMARK repository serves as the **catalog and standards authority**:

```bash
REMARK/
├── REMARKs/                    # Catalog of all REMARKs
│   ├── BufferStockTheory.yml   # Minimal metadata per REMARK
│   ├── beyond-the-streetlight.yml
│   └── ...
├── STANDARD.md                 # Requirements for REMARK compliance
├── cli.py                      # Tools for validation and testing
├── .github/workflows/          # Automation workflows
│   └── transfer-remark-metadata.yml
└── Documentation files
```

#### REMARK Catalog Files (REMARKs/*.yml)

Each REMARK has a minimal YAML file containing:
```yaml
name: project-name              # Short identifier
remote: https://github.com/...  # Repository URL  
title: Human Readable Title     # Display name
```

**Critical Point**: The REMARK repository does NOT contain the full metadata - only the minimal catalog entry pointing to the actual research repository.

### 2. Individual Research Repositories

Each research project is a **self-contained repository** that must meet REMARK standards:

#### Required Files

- **`CITATION.cff`**: Complete bibliographic metadata (CFF format)
- **`REMARK.md`**: Website-specific metadata + abstract content
- **`reproduce.sh`**: Script to reproduce all results
- **`binder/environment.yml`**: Environment specification

#### Optional Files

- **`reproduce_min.sh`**: Quick demonstration version

#### Example REMARK.md Structure

```markdown
---
# Website-specific metadata (YAML frontmatter)
remark-name: beyond-the-streetlight
title-original-paper: "100 years of Economic Measurement..."
notebooks:
  - RS100_Discussion_Slides.ipynb
tags:
  - REMARK
  - Notebook
keywords:
  - forecast accuracy
  - Federal Reserve
---

# Abstract

This repository provides analysis of...
```

### 3. Website Generation Process

**🌐 WEBSITE GENERATION SYSTEM** (Primary: `populate_remarks.py`)

The econ-ark.org website is generated through an **automated pipeline** that is SEPARATE from the REMARK validation system:

#### Step 1: GitHub Workflows

Two workflows coordinate the integration:

**A. REMARK Repo → Website Repo** (`.github/workflows/transfer-remark-metadata.yml`)

- Runs daily at 8:00 AM UTC
- Copies any existing `REMARKs/*.md` files to `econ-ark.org/_materials/`
- **Important**: This is a SECONDARY mechanism for edge cases where manual `.md` files exist
- **Not the primary workflow** - most REMARKs only have `.yml` catalog entries

**B. Website Preprocessing** (`.github/workflows/site-preprocess.yml`) - **PRIMARY MECHANISM**

- Runs on every push to master
- Executes `scripts/populate_remarks.py` (the core integration script)
- This is what actually builds the website content for most REMARKs

#### Step 2: populate_remarks.py Script

This is the **core integration script** that:

1. **Clones REMARK catalog**: Gets the current list of all REMARKs
2. **Reads catalog entries**: Extracts repository URLs from `REMARKs/*.yml` files
3. **Clones individual repositories**: Downloads each research project (using `--sparse` clone)
4. **Merges metadata**: Combines data from two key source files:
   - `CITATION.cff` (bibliographic metadata)
   - `REMARK.md` (website-specific fields + abstract/body content)
5. **Generates material files**: Creates `_materials/{name}.md` for Jekyll

**🚨 IMPORTANT**: This script only requires `CITATION.cff` to generate a basic webpage. For a rich, descriptive page, `REMARK.md` is essential. The script specifically looks for these two file names and ignores other markdown files (e.g., `README.md` or legacy `{name}.md` files) for website content generation.

#### Step 3: Jekyll Site Generation

- Jekyll processes `_materials/*.md` files into web pages
- Templates in `_layouts/` control rendering
- Collections system enables filtering and search

## Data Flow Diagram

```monospace
┌─────────────────────┐
│ Author submits PR   │
│ to REMARK repo      │
│ (adds .yml file)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    REMARK Repository                                │
│                                                                     │
│  REMARKs/new-project.yml ◄─── PR Review & Merge                   │
│  ┌─────────────────────┐                                           │
│  │ name: new-project   │                                           │
│  │ remote: github.com/ │                                           │
│  │ title: Project Name │                                           │
│  └─────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
           │
           ▼ (Daily/Push triggers)
┌─────────────────────────────────────────────────────────────────────┐
│                populate_remarks.py Script                          │
│                                                                     │
│  1. Clone REMARK repo ──► Get catalog                              │
│  2. For each entry:                                                 │
│     ├─ Clone individual repo                                        │
│     ├─ Read CITATION.cff                                           │
│     ├─ Read REMARK.md                                              │
│     └─ Merge metadata                                               │
│  3. Generate _materials/{name}.md                                   │
└─────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     econ-ark.org Website                           │
│                                                                     │
│  _materials/                                                        │
│  ├─ new-project.md  ◄─── Generated from merged metadata            │
│  │   ┌─────────────────────────────────────────────────────────┐   │
│  │   │ ---                                                     │   │
│  │   │ # From CITATION.cff                                     │   │
│  │   │ authors: [...]                                          │   │
│  │   │ title: Project Name                                     │   │
│  │   │ version: 1.0.0                                          │   │
│  │   │ # From REMARK.md frontmatter                            │   │
│  │   │ remark-name: new-project                                │   │
│  │   │ notebooks: [...]                                        │   │
│  │   │ tags: [REMARK, ...]                                     │   │
│  │   │ ---                                                     │   │
│  │   │                                                         │   │
│  │   │ # From REMARK.md body                                   │   │
│  │   │ # Abstract                                              │   │
│  │   │ This repository provides...                             │   │
│  │   └─────────────────────────────────────────────────────────┘   │
│  │                                                                 │
│  └─ Jekyll processes → /materials/new-project/ webpage             │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Integration Points

### Metadata Merging Logic

The `populate_remarks.py` script combines metadata with this priority:

1. **Base data**: `CITATION.cff` provides bibliographic information
2. **Website overlay**: `REMARK.md` frontmatter adds website-specific fields
3. **Content**: `REMARK.md` body becomes the webpage content

### File Naming Convention

- REMARK catalog: `REMARKs/{name}.yml`
- Individual repo: `{name}/REMARK.md` and `{name}/CITATION.cff`
- Website material: `_materials/{name}.md`
- Final URL: `econ-ark.org/materials/{name}/`

### Error Handling

- Missing `CITATION.cff`: Project skipped (no webpage generated)
- Missing `REMARK.md`: Uses only CITATION.cff data
- Invalid YAML: Build fails with error

## Submission Process

### For New REMARKs

1. **Prepare repository** meeting REMARK standards
2. **Submit PR** to REMARK repo adding `REMARKs/{name}.yml`
3. **Editorial review** checks compliance
4. **Merge PR** adds to catalog
5. **Automated integration** generates website content

### For Updates

1. **Update individual repository** (tag new release)
2. **Website auto-updates** within 24 hours via scheduled workflow
3. **Manual trigger** available via GitHub Actions

## Common Issues and Solutions

### Issue: New REMARK not appearing on website

**Diagnosis**: Check if `CITATION.cff` exists and is valid YAML
**Solution**: Ensure all required files are present and properly formatted

### Issue: Metadata not updating

**Diagnosis**: Check GitHub Actions logs for `populate_remarks.py`
**Solution**: Verify individual repository is publicly accessible

### Issue: Website build failure

**Diagnosis**: YAML parsing error in metadata files
**Solution**: Validate YAML syntax in `CITATION.cff` and `REMARK.md`

### Issue: GitHub workflow appears "broken" (looking for .md files when only .yml exist)

**Diagnosis**: Misunderstanding the dual workflow system
**Solution**: Remember that `transfer-remark-metadata.yml` is SECONDARY - the primary workflow is `populate_remarks.py`

## Critical Understanding Points for AI Systems

⚠️ **Warning for AI Analysis**: The REMARK ecosystem uses a **dual workflow system**:

1. **Primary**: `populate_remarks.py` script that reads `.yml` catalog files and generates content
2. **Secondary**: `transfer-remark-metadata.yml` workflow for edge cases with manual `.md` files

**Do not assume the transfer workflow is misconfigured** because it looks for `.md` files in a directory containing `.yml` files. This is by design - the two mechanisms serve different purposes.

## Maintenance and Monitoring

### Automated Processes

- Daily metadata sync (8:00 AM UTC)
- Website rebuild on every push
- Link validation (via GitHub Actions)

### Manual Oversight

- Editorial review of new submissions
- Quality assurance testing
- Compliance checking via CLI tools

---

This workflow ensures that the REMARK ecosystem maintains high standards for reproducibility while providing a seamless integration between distributed research repositories and the centralized discovery platform at econ-ark.org.

## 🔧 REMARK Validation System vs 🌐 Website Generation System

### Critical Distinction

**These are TWO COMPLETELY SEPARATE SYSTEMS with different purposes and requirements:**

| Aspect | 🌐 Website Generation (`populate_remarks.py`) | 🔧 REMARK Validation (`cli.py`) |
|--------|-----------------------------------------------|----------------------------------|
| **Purpose** | Generate econ-ark.org website content | Validate research reproducibility |
| **Trigger** | Automatic (daily/push) | Manual (editor workflow) |
| **Required Files** | `CITATION.cff` (required), `REMARK.md` (optional) | `reproduce.sh`, `CITATION.cff`, `binder/environment.yml` |
| **Clone Method** | `git clone --sparse` (metadata only) | `git clone --depth 1` (full repo) |
| **Output** | `_materials/*.md` files for Jekyll | Validation reports and logs |
| **Failure Impact** | Missing materials on website | Cannot reproduce research |

### 🌐 Website Generation Requirements

**Minimum for website appearance:**

- ✅ `CITATION.cff` - Provides author, title, abstract, etc.
- ✅ Valid repository URL in `REMARKs/*.yml`

**Enhanced website features:**

- ✅ `REMARK.md` - Adds website-specific metadata (notebooks, tags, custom content)

**NOT required for website:**

- ❌ `reproduce.sh`
- ❌ `binder/environment.yml`
- ❌ `reproduce_min.sh`

### 🔧 REMARK Validation Requirements

**Required for REMARK compliance (reproducibility):**

- ✅ `reproduce.sh` - Must run and reproduce all results
- ✅ `CITATION.cff` - Bibliographic metadata
- ✅ `binder/environment.yml` - Environment specification
- ✅ Tagged release

**Optional for REMARK compliance:**

- ✅ `reproduce_min.sh` - Quick demonstration version

### Common Confusion Points

**❌ WRONG**: "If `cli.py lint` shows missing files, the website won't work"
**✅ CORRECT**: Website generation and REMARK validation are independent

**❌ WRONG**: "All materials on the website are REMARK compliant"
**✅ CORRECT**: Materials can appear on website without meeting full REMARK standards

**❌ WRONG**: "Website generation requires reproducibility files"
**✅ CORRECT**: Website only needs bibliographic metadata (`CITATION.cff`)

### Workflow Integration

```bash
Research Repository
├── CITATION.cff           ────► Website Generation (populate_remarks.py)
├── REMARK.md (optional)   ────► Website Enhancement
├── reproduce.sh           ────► REMARK Validation (cli.py)
├── binder/environment.yml ────► REMARK Validation (cli.py)
└── other research files   ────► Research Content
```

### Troubleshooting Guide

**Issue**: Material missing from website
**Check**: Does repository have `CITATION.cff`? Is URL correct in `REMARKs/*.yml`?

**Issue**: REMARK validation fails
**Check**: Does repository have `reproduce.sh`, `binder/environment.yml`? Does `reproduce.sh` run successfully?

**Issue**: Website shows incorrect content
**Check**: Is `CITATION.cff` correct? Does `REMARK.md` have proper frontmatter?

### AI System Warning

**🤖 FOR AI SYSTEMS**: Do not confuse these two systems. When analyzing REMARK compliance issues:

1. **First determine**: Are you looking at website generation issues or reproducibility validation issues?
2. **Website problems**: Check `populate_remarks.py` logs, `CITATION.cff` files, `_materials/` generation
3. **Validation problems**: Check `cli.py` output, `reproduce.sh` scripts, environment files
4. **Remember**: A repository can appear on the website without being fully REMARK compliant for reproducibility

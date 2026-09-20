# Tier 3+ Enhanced: Journal-Specific Requirements

**Date**: December 7, 2025  
**Purpose**: Upgrade from Tier 3 (LCD) to journal-specific standards  
**Audience**: Authors preparing for journals with stricter requirements

---

## Overview

**Tier 3 (LCD)** meets the **minimum** requirements for ALL major economics journals.

Some journals have **MORE stringent** requirements. This guide helps you upgrade from Tier 3 (LCD) to journal-specific standards.

---

## Who Needs This Guide?

### You DON'T need enhancements if:
- Preparing for first submission to most journals
- Journal submission guidelines don't specify additional requirements
- You want baseline "publication ready" status

### You DO need enhancements for:
- **Quantitative Economics (QE)**: Most stringent requirements
- **American Economic Review (AER)**: Specific archival platform
- Journals explicitly requiring specific data formats or structures
- Maximum reproducibility and long-term preservation

---

## Enhancement Checklist by Journal

### Quantitative Economics (QE)

QE has the **most stringent** replication requirements in economics.

#### Additional Requirements Beyond Tier 3 LCD:

**1. Zenodo DOI (REQUIRED)**
- [ ] Create GitHub release
- [ ] Enable Zenodo-GitHub integration
- [ ] Submit to QE Zenodo community
- [ ] Add DOI to CITATION.cff
- [ ] Add DOI badge to README

**Time**: 30 minutes

**2. 7-Section README Structure (REQUIRED)**

Must have these specific sections:
- [ ] Data Availability Statement (separate, dedicated section)
- [ ] Computational Requirements (exact versions)
- [ ] Installation Instructions
- [ ] Reproduction Instructions
- [ ] Code Organization
- [ ] Output Mapping (code → tables/figures)
- [ ] Data Citations (dedicated section with full bibliographic format)

**Time**: 1-2 hours to reorganize

**3. CSV Data (REQUIRED)**
- [ ] Convert all data to CSV/TXT/JSON
- [ ] Proprietary formats may supplement but cannot replace
- [ ] Variable codebooks required
- [ ] Data citations in bibliographic format

**Time**: 30-60 minutes

**4. Exact Version Specification (REQUIRED)**
- [ ] Exact Python version (3.9.6, not 3.9)
- [ ] Exact versions for all critical packages
- [ ] Document in README Computational Requirements section

**Time**: 15 minutes

**5. Output Mapping Table (REQUIRED)**
- [ ] Create explicit table: code file → table/figure number
- [ ] Include line numbers where possible
- [ ] Format as table in README

**Time**: 30 minutes

**Total Time**: ~3-4 hours to upgrade from Tier 3 LCD to QE compliance

---

### American Economic Review (AER)

#### Additional Requirements Beyond Tier 3 LCD:

**1. openICPSR Archive (REQUIRED)**
- [ ] Create openICPSR repository (not Zenodo)
- [ ] Follow AEA Data Editor's README template
- [ ] DOI issued by AEA

**Note**: Different platform than Zenodo

**Time**: 1 hour

**2. Social Science Data Editors Template (RECOMMENDED)**
- [ ] Use specific section structure
- [ ] Data Availability Statement (AEA format)
- [ ] Computational requirements section

**Time**: 1 hour

**3. Data Citation Format (REQUIRED)**
- [ ] Follow AEA bibliographic format
- [ ] Include data citations in dedicated section

**Time**: 30 minutes

**Total Time**: ~2-3 hours

---

### Econometrica (ECMA)

#### Additional Requirements Beyond Tier 3 LCD:

**1. Econometric Society Archive (REQUIRED)**
- [ ] Submit to Econometric Society platform (not Zenodo/openICPSR)
- [ ] Follow ECMA-specific guidelines

**Time**: 1 hour

**2. Replication Report (REQUIRED)**
- [ ] Create replication report documenting all steps
- [ ] Include runtime estimates
- [ ] Platform testing results

**Time**: 1-2 hours

**Total Time**: ~2-3 hours

---

### Other Top Journals

Most other top journals (REStud, JPE, QJE, REStat) accept Tier 3 LCD with minor additions:

#### Typical Additions:
- Recommended: Data Availability Statement (can be in README)
- Recommended: Platform-specific hosting (if specified)
- Required: Follow journal template (if provided)

**Total Time**: Usually <1 hour

---

## Enhancement Priority Guide

### Essential (Do First)

1. **Platform-specific archival** (if required by journal)
   - QE → Zenodo
   - AER → openICPSR
   - ECMA → Econometric Society

2. **README structure** (if journal specifies)
   - Reorganize to match template
   - Add required sections

### Important (Do Second)

3. **Data format conversion** (if required)
   - Convert to CSV
   - Add codebooks
   - Update documentation

4. **Exact version pinning** (if required)
   - Document exact patch versions
   - Create lockfile if needed

### Nice to Have (Do Third)

5. **Output mapping table** (if required)
   - Create explicit mapping
   - Add line numbers

6. **Multiple platform testing** (if required)
   - Test on additional platforms
   - Document differences

---

## Conversion Scripts

### Stata to CSV Conversion

```bash
#!/bin/bash
# Convert all .dta files to .csv

for file in data/*.dta; do
    basename="${file%.dta}"
    stata -b "use \"$file\", clear
              export delimited \"${basename}.csv\", replace"
done
```

### Environment Lockfile Creation

```bash
# Create exact version lockfile from current environment
conda env export --from-history > environment-exact.yml

# Or for pip
pip freeze > requirements-exact.txt
```

---

## README Template: QE-Specific Enhancements

### Data Availability Statement (QE Format)

```markdown
## Data Availability Statement

### Primary Data Source

**Dataset Name**: Survey of Consumer Finances (SCF) 2019  
**Publisher**: Board of Governors of the Federal Reserve System  
**DOI/URL**: https://www.federalreserve.gov/econres/scfindex.htm  
**Access**: Public, free download  
**Date Accessed**: 2023-09-15  
**Version**: Public release version (2020-05-12)

**Citation**:
Board of Governors of the Federal Reserve System. (2020). 
Survey of Consumer Finances, 2019 [Data set]. 
Federal Reserve Board. 
https://www.federalreserve.gov/econres/scfindex.htm

### Data Included in Repository

**File**: `data/scf_processed.csv`  
**Source**: Derived from SCF 2019 (cited above)  
**Format**: CSV (plain text)  
**Size**: 2.3 MB  
**Variables**: See `data/codebook_scf.txt` (includes variable definitions, units, sources)  
**Transformations**: Cleaning code in `code/01_clean_scf.py`

### Restricted Data

This analysis does not use restricted data.
```

### Computational Requirements (QE Format)

```markdown
## Computational Requirements

### Software

**Primary Language**: Python 3.9.6 (exact)

**Critical Packages** (exact versions):
- econ-ark==0.14.1
- numpy==1.24.3
- pandas==2.0.2
- scipy==1.10.1
- matplotlib==3.7.1

**Complete Environment**: See `binder/environment.yml` for full dependency list

### Hardware

**Minimum**: 8GB RAM, 10GB disk space, single CPU core  
**Recommended**: 16GB RAM, 50GB disk space, 4+ CPU cores  
**Used for Paper**: 32GB RAM, 12 CPU cores, 100GB SSD

### Runtime

| Task | Time (Single Core) | Time (4 Cores) |
|------|-------------------|----------------|
| Data preparation | 5 min | 5 min |
| Main estimation | 8 hours | 2 hours |
| Figures/tables | 10 min | 10 min |
| **Full reproduction** | **~8 hours** | **~2 hours** |

### Platforms Tested

- ✅ macOS 12.6 (Apple M1, ARM64) - Results identical
- ✅ Ubuntu 20.04 LTS (x86_64) - Results identical
- ✅ Windows 11 with WSL2 (x86_64) - Results identical

**Numerical Differences**: None expected. Random seed fixed for all stochastic simulations.
```

### Output Mapping (QE Format)

```markdown
## Output Mapping

### Tables

| Paper Table | Output File | Generating Code | Lines |
|-------------|-------------|-----------------|-------|
| Table 1: Summary Statistics | `output/tables/table1.csv` | `code/03_tables.py` | 45-67 |
| Table 2: Main Results | `output/tables/table2.csv` | `code/03_tables.py` | 89-145 |
| Table 3: Robustness | `output/tables/table3.csv` | `code/03_tables.py` | 167-223 |
| Table A1: Appendix | `output/tables/tableA1.csv` | `code/03_tables.py` | 245-289 |

### Figures

| Paper Figure | Output File | Generating Code | Lines |
|--------------|-------------|-----------------|-------|
| Figure 1: Time Series | `output/figures/fig1.pdf` | `code/04_figures.py` | 34-78 |
| Figure 2: Comparison | `output/figures/fig2.pdf` | `code/04_figures.py` | 101-156 |
| Figure 3: Sensitivity | `output/figures/fig3.pdf` | `code/04_figures.py` | 178-234 |

### In-Text Numbers

| Paper Reference | Value Source | Code Location |
|----------------|--------------|---------------|
| Page 12, paragraph 2 | `output/results/main.json:coefficient_beta` | `code/02_analysis.py:456` |
| Page 15, Table 2 note | `output/results/diagnostics.csv:row_3` | `code/02_analysis.py:523` |
```

---

## Time Budget Summary

### From Tier 3 LCD to Journal-Specific

| Target Journal | Additional Time Required |
|---------------|-------------------------|
| QE (Quantitative Economics) | 3-4 hours |
| AER (American Economic Review) | 2-3 hours |
| ECMA (Econometrica) | 2-3 hours |
| Most other top journals | <1 hour |

### Cost-Benefit Analysis

**Tier 3 LCD**: 8-12 hours total effort
- ✅ Meets minimum for ALL journals
- ✅ Good enough for most submissions
- ⚠️ May need enhancements for strictest journals

**Tier 3 + QE Enhancements**: 12-16 hours total
- ✅ Meets ALL journal requirements
- ✅ Maximum reproducibility
- ✅ Best long-term preservation
- ⚠️ More upfront work

---

## Recommendations

### Strategy 1: Start with LCD, Enhance Later
1. Complete Tier 3 (LCD) first → 8-12 hours
2. Submit to journal
3. If journal requires enhancements, add them → 1-4 hours

**Best for**: First submission, uncertain about journal choice

### Strategy 2: Build to Enhanced Immediately
1. Complete Tier 3 (LCD) → 8-12 hours
2. Add QE enhancements → 3-4 hours
3. Total: 12-16 hours upfront

**Best for**: Targeting QE or want maximum reproducibility

### Strategy 3: Journal-Specific Build
1. Complete Tier 3 (LCD) → 8-12 hours
2. Add only required enhancements for target journal → 1-4 hours

**Best for**: Know target journal, want minimal extra work

---

## Frequently Asked Questions

**Q: Do I need Zenodo DOI for AER?**  
A: No, AER uses openICPSR (their own platform).

**Q: If I get a Zenodo DOI, does that satisfy QE?**  
A: Mostly yes, but also submit to QE Zenodo community specifically.

**Q: Can I meet QE requirements and use that for other journals?**  
A: Yes! QE is most stringent, so QE compliance → universal compliance.

**Q: How much do requirements change over time?**  
A: Tier 3 LCD is stable (minimum hasn't changed much). Enhanced requirements may evolve as journals update policies.

**Q: Is it worth doing enhancements before submission?**  
A: Depends on journal and timeline. LCD is usually sufficient for initial submission. Enhancements can be added during revisions if requested.

---

## Journal Contact Information

### For Latest Requirements

- **Quantitative Economics**: https://www.econometricsociety.org/publications/quantitative-economics/information-authors
- **American Economic Review**: https://www.aeaweb.org/journals/policies/data-availability-policy
- **Econometrica**: https://www.econometricsociety.org/publications/econometrica/information-authors

### Data Editors

Most journals now have Data Editors who handle replication. Check journal website for:
- Latest templates
- Specific requirements
- Data Editor contact for questions

---

## Version History

### v1.0 (December 2025)
- Initial guide for Tier 3 LCD enhancement
- QE, AER, ECMA requirements documented

---

**Summary**: Tier 3 (LCD) gets you 90% of the way for all journals. Enhancements add the final 10% for specific journal requirements.

**Recommendation**: Start with Tier 3 (LCD). Add enhancements if/when required by specific journal.

---
# CFF required fields
cff-version: "1.1.0" # required 
message: "For a consumption/saving problem with transitory and permanent shocks and unbounded (CRRA) utility, this paper derives conditions under which a nondegenerate solution exists, and under which a target wealth ratio exists; all results are paired with illustrative numerical solutions." # required
authors: # required
  -
    family-names: "Velásquez-Giraldo"
    given-names: "Mateo"
    orcid: 0000-0001-7243-6776
  -
    family-names: "Zahn"
    given-names: "Matthew"
    orcid: 0000-0003-1148-2036
title: "REMARK: Consumption and Portfolio Choice Over the Life Cycle" # required
abstract: "This REMARK is an attempt to reproduce the main results of Cocco, Gomes, & Maenhout (2005), 'Consumption and Portfolio Choice Over the Life Cycle' (https://academic.oup.com/rfs/article-abstract/18/2/491/1599892)" # abstract: optional
date-released: 2020-08-13 # required

# REMARK required fields
remark-version: 1.0 # required
references: # Formatted metadata of original paper, from BibTex
  - type: article
    authors: # required
      -
        family-names: "Cocco"
        given-names: "Joao F."
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
      -
        family-names: "Gomes"
        given-names: "Francisco J."
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
      -
        family-names: "Maenhout"
        given-names: "Pascal J."
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
    title: "Consumption and Portfolio Choice over the Life Cycle" # required
    doi: "https://doi.org/10.1093/rfs/hhi017" # optional
    date: 2005-02-10
    publisher : "The Review of Financial Studies"

# Econ-ARK website fields
github_repo_url: https://github.com/econ-ark/CGMPortfolio # required 
remark-name: CGMPortfolio # required 
notebooks: # path to any notebooks within the repo - optional
  - 
    Code/Python/CGMPortfolio.ipynb
identifiers-paper: # required for Replications; optional for Reproductions
   - 
      type: url 
      value: https://academic.oup.com/rfs/article-abstract/18/2/491/1599892
   - 
      type: doi
      value: https://doi.org/10.1093/rfs/hhi017
date-published-original-paper: 2005-02-10 # required for Replications; optional for Reproductions

tags: # Use the relavent tags
  - REMARK

identifiers: # optional
  - 
    type: url
    value: "https://github.com/econ-ark/CGMPortfolio"

keywords: # optional
  - Portfolio
  - Risky Assets
  - Consumption
  - Life Cycle
---

# CGMPortfolio

This REMARK is an attempt to reproduce the main results of Cocco, Gomes, & Maenhout (2005), 'Consumption and Portfolio Choice Over the Life Cycle' (https://academic.oup.com/rfs/article-abstract/18/2/491/1599892)

## References

Cocco, J. F., Gomes, F. J., & Maenhout, P. J. (2005). Consumption and portfolio choice over the life cycle. The Review of Financial Studies, 18(2), 491-533.

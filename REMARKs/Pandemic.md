---
# CFF required fields
cff-version: "1.1.0" # required (don't change)
message: "To predict the eﬀects of the 2020 U.S. CARES Act on consumption, we extend a model that matches responses of households to past consumption stimulus packages; all results are paired with illustrative numerical solutions." # required
authors: # required
  -
    family-names: "Carroll"
    given-names: "Christopher D."
    orcid: "https://orcid.org/0000-0003-3732-9312"
  -
    family-names: "Crawley"
    given-names: "Edmund"
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
  -
    family-names: "Slacalek"
    given-names: "Jiri"
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
  -
    family-names: "White"
    given-names: "Matthew N."
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
title: "Modeling the Consumption Response to the CARES Act" # required
abstract: "To predict the eﬀects of the 2020 U.S. CARES Act on consumption, we extend a model that matches responses of households to past consumption stimulus packages. The extension allows us to account for two novel features of the coronavirus crisis. First, during the lockdown, many types of spending are undesirable or impossible. Second, some of the jobs that disappear during the lockdown will not reappear when it is lifted. We estimate that, if the lockdown is short-lived, the combination of expanded unemployment insurance beneﬁts and stimulus payments should be suﬃcient to allow a swift recovery in consumer spending to its pre-crisis levels. If the lockdown lasts longer, an extension of enhanced unemployment beneﬁts will likely be necessary if consumption spending is to recover." # abstract: optional

# REMARK required fields
remark-version: "1.0" # required
references: # required for replications; optional for reproductions; BibTex data from original paper
  - type: article
    authors: # required
      -
        family-names: "Carroll"
        given-names: "Christopher D."
        orcid: "https://orcid.org/0000-0003-3732-9312"
      -
        family-names: "Crawley"
        given-names: "Edmund"
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
      -
        family-names: "Slacalek"
        given-names: "Jiri"
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
      -
        family-names: "White"
        given-names: "Matthew N."
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
    title: "Modeling the Consumption Response to the CARES Act" # required
    doi: "https://doi.org/10.3386/w27876" # optional
    date: 2020-09-14 # required
    publisher: "NBER"

# Econ-ARK website fields
github_repo_url: https://github.com/econ-ark/Pandemic # required 
remark-name: Pandemic # required 
dashboards: # path to any dahsboards within the repo - optional
  - 
    Code/Python/dashboard.ipynb

tags: # Use the relavent tags
  - REMARK
  - Notebook

keywords: # optional
  - Consumption
  - COVID-19
  - Stimulus
  - Fiscal Policy
---

# Pandemic-Consumption-Response

[![badge](https://img.shields.io/badge/Launch-Dashboard-579ACA.svg?logo=data:image/png;base64,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)](https://xhrtcvh6l53u.curvenote.dev/services/binder/v2/gh/econ-ark/Pandemic/HEAD?urlpath=/voila/render/Code/Python/dashboard.ipynb)

This repository is a complete software archive for the paper "Modeling the Consumption Response to the CARES Act" by Carroll, Crawley, Slacalek, and White (2020). This README file provides instructions for running our code on your own computer, as well as adjusting the parameters of the model to produce alternate versions of the figures in the paper.

## References

Carroll, C. D., Crawley, E., Slacalek, J., & White, M. N. (2020). Modeling the consumption response to the CARES Act (No. w27876). National Bureau of Economic Research.

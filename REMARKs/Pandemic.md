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

# Pandemic-ConsumptionResponse

This repository is a complete software archive for the paper "Modeling the Consumption Response to the CARES Act" by Carroll, Crawley, Slacalek, and White (2020). This README file provides instructions for running our code on your own computer, as well as adjusting the parameters of the model to produce alternate versions of the figures in the paper.

## References

Carroll, C. D., Crawley, E., Slacalek, J., & White, M. N. (2020). Modeling the consumption response to the CARES Act (No. w27876). National Bureau of Economic Research.

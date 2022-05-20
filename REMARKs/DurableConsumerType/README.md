---
tags:
  - REMARK
  - Notebook
authors:
  -
    family-names: Monninger
    given-names: Adrian B. 
    orcid: "https://orcid.org/0000-0001-5978-5621"
cff-version: "1.1.0"
abstract: "Druedahl (2021) Replication. Druedahl solves a consumer saving model with durable and non-durable goods. 
This notebook replicates his nested endogeneous grid method (nested EGM) extended with an upper envelope step. Note that
while his codes have persistent income as an additional state variable, this replication only uses variables normalized by
permanent income."
title: "DurableConsumerType"
version: "1.0"
# REMARK fields
#github_repo_url: https://github.com/AMonninger/REMARK-DurableConsumerType
notebooks: # path to any notebooks within the repo - optional
  - 
    Code/Python/DurableModel_Notebook.ipynb
remark-name: DurableConsumerType # required 
reference: # required for Replications; optional for Reproductions
   - title: "A Guide on Solving Non-convex Consumption-Saving Models"
   - 
      type: doi
      value: https://doi.org/10.1007/s10614-020-10045-x
   - authors:
        -
          family-names: Druedahl
          given-names: Jeppe
---



# KrusellSmith

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/DurableConsumerType/HEAD)

This is a Replication of Druedahl, 2021.


## References

Druedahl, J. (2021). A guide on solving non-convex consumption-saving models. Computational Economics, 58(3), 747-775.

---
# CFF required fields
cff-version: "1.1.0" # required (don't change)
authors: # required
  -
    family-names: "Velásquez-Giraldo"
    given-names: "Mateo"
    orcid: https://orcid.org/0000-0001-7243-6776
title: "A Two-Asset Savings Model with an Income-Contribution Scheme" # required
abstract: "This paper develops a two-asset consumption-savings model and serves as
the documentation of an open-source implementation of methods to solve and
simulate it in the HARK toolkit. The model represents an agent who can
save using two different assets---one risky and the other risk-free---to insure
against fluctuations in his income, but faces frictions to transferring funds between
assets. The flexibility of its implementation and its inclusion in the HARK
toolkit will allow users to adapt the model to realistic life-cycle calibrations, and
also to embedded it in heterogeneous-agents macroeconomic models." # optional
date-released: 2021-06-17 # required

# REMARK required fields
remark-version: "v1.0.1" # required
references: # required for replications; optional for reproductions; BibTex data from original paper
  - type: article
    authors: # required
      -
        family-names: "Velásquez-Giraldo"
        given-names: "Mateo"
        orcid: https://orcid.org/0000-0001-7243-6776
      -
        family-names: "Author 2 Last Name"
        given-names: "Author 2 First Name"
        orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
    title: "A Two-Asset Savings Model with an Income-Contribution Scheme" # required
    doi: "https://zenodo.org/badge/DOI/10.5281/zenodo.4974234.svg" # optional

# Econ-ARK website fields
github_repo_url: https://github.com/econ-ark/RiskyContrib # required
remark-name: RiskyContrib # required
notebooks: # path to any notebooks within the repo - optional
  -
    Code/Python/RiskyContrib.ipynb

tags: # Use the relavent tags
  - REMARK

keywords: # optional
  - Lifecycle
  - Portfolio Choice
  - Social Security
  - Open Source

---

# A Two-Asset Savings Model with an Income-Contribution Scheme

This paper develops a two-asset consumption-savings model and serves as
the documentation of an open-source implementation of methods to solve and
simulate it in the HARK toolkit. The model represents an agent who can
save using two different assets---one risky and the other risk-free---to insure
against fluctuations in his income, but faces frictions to transferring funds between
assets. The flexibility of its implementation and its inclusion in the HARK
toolkit will allow users to adapt the model to realistic life-cycle calibrations, and
also to embedded it in heterogeneous-agents macroeconomic models.

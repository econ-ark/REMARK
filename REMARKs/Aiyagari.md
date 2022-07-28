---
# CFF required fields
cff-version: 1.1.0 # required (don't change)
authors: # required
  -
    family-names: Huang
    given-names: Zixuan 
title: AiyagariIdiosyncratic # required
abstract: Aiyagari (1994) Replication # optional

# REMARK required fields
remark-version: 1.0 # required - specify version of REMARK standard used
reference: # required for replications; optional for reproductions; BibTex data from original paper
  - type: article
    authors: # required
      -
        family-names: Krusell
        given-names: Per
      -
        family-names: "Smith, Jr."
        given-names: "Anthony A."
    title: "Income and Wealth Heterogeneity in the Macroeconomy"
    doi: https://doi.org/10.1086/250034
    date: 1998
    publisher: Journal of political Economy

# Econ-ARK website fields
github_repo_url: https://github.com/econ-ark/KrusellSmith # required 	
remark-name: KrusellSmith # required 
notebooks: # path to any notebooks within the repo - optional
  - 
    Code/Python/KrusellSmith.ipynb

tags:
  - REMARK
  - Notebook
---



# KrusellSmith

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/KrusellSmith/HEAD)

This is a Replication of Krusell and Smith, 1998.


## References

Krusell, P., & Smith, Jr, A. A. (1998). Income and wealth heterogeneity in the macroeconomy. Journal of political Economy, 106(5), 867-896.

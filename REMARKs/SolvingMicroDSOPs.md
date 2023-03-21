---
# CFF required fields
cff-version: "1.1.0" # required
authors: # required
  -
    family-names: "Carroll"
    given-names: "Christopher D."
    orcid: "https://orcid.org/0000-0003-3732-9312"
  -
    family-names: "Wang"
    given-names: "Tao"
    orcid: "https://orcid.org/0000-0003-4806-8592"
title: "Solution Methods for Microeconomic Dynamic Stochastic Optimization Problems" # required
abstract: "These notes describe tools for solving microeconomic dynamic stochastic optimization problems, and show how to use those tools for efficiently estimating a standard life cycle consumption/saving model using microeconomic data.  No attempt is made at a systematic overview of the many possible technical choices; instead, I present a specific set of methods that have proven useful in my own work (and explain why other popular methods, such as value function iteration, are a bad idea).  Paired with these notes is Mathematica, Matlab, and Python software that solves the problems described in the text." # abstract: optional
date-released: 2021-02-20 # required

# REMARK required fields
remark-version: "1.0" # required
references:
  - type: lecture-notes
    authors: # required
      -
        family-names: "Carroll"
        given-names: "Christopher D."
        orcid: "https://orcid.org/0000-0003-3732-9312"
    title: "Solution Methods for Microeconomic Dynamic Stochastic Optimization Problems"
    repository: "https://github.com/llorracc/SolvingMicroDSOPs" # optional

# Econ-ARK website fields
github_repo_url: https://github.com/econ-ark/SolvingMicroDSOPs # required
remark-name: SolvingMicroDSOPs # required
notebooks: # path to any notebooks within the repo - optional
  - SolvingMicroDSOPs.ipynb

identifiers: # optional
  -
    type: url
    value: "https://llorracc.github.io/SolvingMicroDSOPs"

tags:
  - REMARK
  - Replication
  - Teaching
  - Tutorial

keywords: # optional
  - Consumption
  - Saving
---

# Solution Methods for Microeconomic Dynamic Stochastic Optimization Problems

These notes describe tools for solving microeconomic dynamic stochastic optimization problems, and show how to use those tools for efficiently estimating a standard life cycle consumption/saving model using microeconomic data.  No attempt is made at a systematic overview of the many possible technical choices; instead, I present a specific set of methods that have proven useful in my own work (and explain why other popular methods, such as value function iteration, are a bad idea).  Paired with these notes is Mathematica, Matlab, and Python software that solves the problems described in the text.

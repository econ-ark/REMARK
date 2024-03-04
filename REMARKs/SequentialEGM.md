---
# CFF required fields
cff-version: 1.2.0 
message: "If you use this software, please cite it as below."
authors:
  - family-names: "Lujan"
    given-names: "Alan"
title: "EGM$^n$: The Sequential Endogenous Grid Method"
abstract: |
     Heterogeneous agent models with multiple decisions are often
     solved using inefficient grid search methods that require many
     evaluations and are slow. This paper provides a novel method for
     solving such models using an extension of the Endogenous Grid
     Method (EGM) that uses Gaussian Process Regression (GPR) to
     interpolate functions on unstructured grids. First, I propose an
     intuitive and strategic procedure for decomposing a problem into
     subproblems which allows the use of efficient solution methods.
     Second, using an exogenous grid of post-decision states and
     solving for an endogenous grid of pre-decision states that obey a
     first-order condition greatly speeds up the solution process.
     Third, since the resulting endogenous grid can often be
     non-rectangular at best and unstructured at worst, GPR provides
     an efficient and accurate method for interpolating the value,
     marginal value, and decision functions. Applied sequentially to
     each decision within the problem, the method is able to solve
     heterogeneous agent models with multiple decisions in a fraction
     of the time and with less computational resources than are
     required by standard methods currently used. Software to
     reproduce these methods is available under the
     https://econ-ark.org/ project for the python programming
     language.


# REMARK required fields
remark-version: 1.0 # required - specify version of REMARK standard used

# Econ-ARK website fields
github_repo_url: https://github.com/alanlujan91/SequentialEGM
remark-name: "SequentialEGM"
notebooks: 
  - code/EGMN/example_ConsPensionModel.ipynb

tags: # Use the relavent tag
  - REMARK
  - Notebook

identifiers-paper:
   - type: url 
     value: https://alanlujan91.github.io/SequentialEGM/egmn/
---

# EGM$^n$ The Sequential Endogenous Grid Method"

[![badge](https://img.shields.io/badge/Launch-Curvenote-E66581.svg?logo=data:image/png;base64,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)](https://josecon.curve.space/articles/018dc338-e64c-7c68-9b37-4f3d092f4252)

Heterogeneous agent models with multiple decisions are often
solved using inefficient grid search methods that require many
evaluations and are slow. This paper provides a novel method for
solving such models using an extension of the Endogenous Grid
Method (EGM) that uses Gaussian Process Regression (GPR) to
interpolate functions on unstructured grids. First, I propose an
intuitive and strategic procedure for decomposing a problem into
subproblems which allows the use of efficient solution methods.
Second, using an exogenous grid of post-decision states and
solving for an endogenous grid of pre-decision states that obey a
first-order condition greatly speeds up the solution process.
Third, since the resulting endogenous grid can often be
non-rectangular at best and unstructured at worst, GPR provides
an efficient and accurate method for interpolating the value,
marginal value, and decision functions. Applied sequentially to
each decision within the problem, the method is able to solve
heterogeneous agent models with multiple decisions in a fraction
of the time and with less computational resources than are
required by standard methods currently used. Software to
reproduce these methods is available under the
<https://econ-ark.org/> project for the python programming
language.

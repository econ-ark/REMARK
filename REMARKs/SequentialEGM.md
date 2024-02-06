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

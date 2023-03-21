---
# CFF required fields
cff-version: "1.1.0" # required 
message: "This paper shows that liquidity constraints and precautionary saving are closely related to each other, since both can be thought of is \"counterclockwise concavifications\" of the consumption function.; all results are paired with illustrative numerical solutions." # required
authors: # required
  -
    family-names: "Carroll"
    given-names: "Christopher D."
    orcid: "https://orcid.org/0000-0003-3732-9312"
  -
    family-names: "Kimball"
    given-names: "Miles S."
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
  -
    family-names: "Holm"
    given-names: "Martin B."
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
title: "Liquidity Constraints and Precautionary Saving" # required
abstract: "We provide the analytical explanation of strong interactions between precautionary sav- ing and liquidity constraints that are regularly observed in numerical solutions to consump- tion/saving models. The effects of constraints and of uncertainty spring from the same cause: concavification of the consumption function, which can be induced either by constraints or by uncertainty. Concavification propagates back to consumption functions in prior periods. But, surprisingly, once a linear consumption function has been concavified by the presence of either risks or constraints, the introduction of additional concavifiers in a given period can reduce the precautionary motive in earlier periods at some levels of wealth." # abstract: optional
date-released: 2020-09-14 # required

# REMARK required fields
remark-version: "1.0" # required - specify version of REMARK standard used
references: # required for replications; optional for reproductions; BibTex data from original paper
  - type: article
    authors: # required
      -
        family-names: "Carroll"
        given-names: "Christopher D."
        orcid: "https://orcid.org/0000-0003-3732-9312"
      -
        family-names: "Kimball"
        given-names: "Miles S."
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
      -
        family-names: "Holm"
        given-names: "Martin B."
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
    title: "Liquidity Constraints and Precautionary Savings" # required
    doi: "https://doi.org/10.1016/j.jet.2021.105276" # optional
    date: 2021-07 # required
    publisher: "Journal of Economic Theory" #optional

# Econ-ARK website fields
github_repo_url: https://github.com/econ-ark/LiqConstr # required 
remark-name: LiqConstr # required 
notebooks: # path to any notebooks within the repo - optional
  - 
    LiqConstr.ipynb
dashboards: # path to any dahsboards within the repo - optional
  - 
    LiqConstr-Dashboard.ipynb
tags: # Use the relavent tags
  - REMARK
  - Reproduction

keywords: # optional
  - liquidity constraints
  - uncertainty
  - precautionary saving

---

# Liquidity Constraints and Precautionary Saving

The LiqConstr directory contains code to reproduce the figures of the paper [Liquidity Constraints and Precautionary Saving](http://econ.jhu.edu/people/ccarroll/papers/LiqConstr/) by Carroll, Holm, and Kimball,
and the LaTeX source to produce the paper once the figures have been created.

## References

Carroll, C. D., Holm, M. B., & Kimball, M. S. (2021). Liquidity constraints and precautionary saving. Journal of Economic Theory, 195, 105276.

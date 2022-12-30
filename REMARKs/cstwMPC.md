---
# CFF required fields
cff-version: 1.1.0 # required (don't change)
message: The results of this paper are now generated in an updated repository using a more modern version of HARK, https://github.com/econ-ark/DistributionOfWealthMPC. Any followup work should build on the updated code in that repository. # optional
authors: # required
  -
    family-names: "Carroll"
    given-names: "Christopher D."
    orcid: "https://orcid.org/0000-0003-3732-9312"
  -
    family-names: "Slacalek"
    given-names: "Jiri"
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
  -
    family-names: "Kiichi"
    given-names: "Tokuoka"
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
  -
    family-names: "White"
    given-names: "Matthew"
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
title: DistributionOfWealthMPC

# REMARK required fields
remark-version: 2.0
references: 
  - type: article
    authors: # required
      -
        family-names: "Carroll"
        given-names: "Christopher D."
        orcid: "https://orcid.org/0000-0003-3732-9312"
      -
        family-names: "Slacalek"
        given-names: "Jiri"
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
      -
        family-names: "Kiichi"
        given-names: "Tokuoka"
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
      -
        family-names: "White"
        given-names: "Matthew N."
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
    title: "The distribution of wealth and the marginal propensity to consume" # required
    doi: "https://doi.org/10.3982/QE694"
    date: 2017-11-20
    publisher: Quantitative Economics

# Econ-ARK website fields
github_repo_url: https://github.com/econ-ark/DistributionOfWealthMPC
remark-name: DistributionOfWealthMPC

tags: # Use the relavent tags
  - REMARK
  - Replication
---


# DistributionOfWealthMPC 
The results of this paper are now generated in an updated repository using a more modern version of HARK, [DistributionOfWealth](https://github.com/econ-ark/DistributionOfWealth). Any followup work should build on the updated code in that repository.

The main results in that paper were generated using Mathematica code that can be executed by running from the command line 

`./Code/Mathematica/DoAll.m`

or by opening the corresponding notebook file `./Code/Matheamtica/DoAll.nb`

The Mathematica results have been replicated using the HARK toolkit.  See the README.md file in the repo for details.

## References

Carroll, C., Slacalek, J., Tokuoka, K., & White, M. N. (2017). The distribution of wealth and the marginal propensity to consume. Quantitative Economics, 8(3), 977-1020.


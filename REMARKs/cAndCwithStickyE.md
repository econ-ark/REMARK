---
# CFF required fields
cff-version: 1.1.0 # required (don't change)
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
    family-names: "Kiichi"
    given-names: "Tokuoka"
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
  -
    family-names: "White"
    given-names: "Matthew"
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
title: "Sticky Expectations and Consumption Dynamics" # required
abstract: "Sticky Expectations and Consumption Dynamics." # abstract: optional

# REMARK required fields
remark-version: 1.0
references:
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
        family-names: "Kiichi"
        given-names: "Tokuoka"
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
      -
        family-names: "White"
        given-names: "Matthew N."
        # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
    title: "Sticky Expectations and Consumption Dynamics" # required
    doi: "https://doi.org/10.1257/mac.20180286" # optional
    date: 2020-07 # required
    publisher: "American Economic Journal: Macroeconomics"

# Econ-ARK website fields
github_repo_url: https://github.com/econ-ark/cAndCwithStickyE # required
remark-name: cAndCwithStickyE # required

tags: # Use the relavent tags
  - REMARK
  - Reproduction

---
# cAndCwithStickyE is a Reproduction

This is a reproduction of the results in the paper "Sticky Expectations and Consumption Dynamics" by Carroll, Crawley, Slacalek, Tokuoka, and White. The [html version](http://econ.jhu.edu/people/ccarroll/papers/cAndCwithStickyE) contains links to resources including the PDF version, the presentation slides, a repo containing the paper and code, and related material

The root directory contains three files:

* `do_min.py` executes the representative agent version of the model
* `do_mid.py` reproduces some of the core results of the paper
* `do_all.py` reproduces all computational results in the paper

Any of these can be run using ipython from the command line, for example:

ipython do_min.py

In all cases, there should be a message that indicates how long it takes to run the code on a particular computer with a particular configuration.

## References

Carroll, C. D., Crawley, E., Slacalek, J., Tokuoka, K., & White, M. N. (2020). Sticky expectations and consumption dynamics. American economic journal: macroeconomics, 12(3), 40-76.


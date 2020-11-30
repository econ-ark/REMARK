---
tags:
  - REMARK
  - Reproduction
abstract: "Sticky Expectations and Consumption Dynamics." # abstract: optional
authors: # required
  -  
    family-names: Carroll
    given-names: "Christopher D."
    orcid: "https://orcid.org/0000-0003-3732-9312"
  -
    family-names: Crawley
    given-names: "Edmund"
    orcid: "https://orcid.org/"
  -
    family-names: Slacalek
    given-names: "Jiri"
    orcid: "https://orcid.org/"
  -
    family-names: Kiichi
    given-names: "Tokuoka"
    orcid: "https://orcid.org/"
  -
    family-names: White
    given-names: "Matthew"
    orcid: "https://orcid.org/"
cff-version: "" # required 
date-released:  # required
identifiers: # optional
  - 
    type: url
    value: "https://github.com/llorracc/cAndCwithStickyE"
  - 
    type: doi
    value: "TODO"
keywords: # optional
message: "" # required
version: "" # required
# REMARK fields
github_repo_url: https://github.com/llorracc/cAndCwithStickyE # required 
commit: # Git commit number that the REMARK will always use; required for "frozen" remarks, optional for "draft" remarks
remark-name: cAndCwithStickyE # required 
title-original-paper: # optional 
notebooks: # path to any notebooks within the repo - optional
dashboards: # path to any dahsboards within the repo - optional
identifiers-paper: # required for Replications; optional for Reproductions
   - 
      type: url 
      value: 
   - 
      type: doi
      value: 
date-published-original-paper: # required for Replications; optional for Reproductions
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


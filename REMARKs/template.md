---
# CFF Requires fileds
cff-version: 1.1.0
message: If you use this software, please cite it as below.
authors:
  - family-names: Holmes
    given-names: Mycroft
title: My Research Tool
version: 1.0.4 # Version of the software released
date-released: 2017-12-18

# REMARK required fields
remark-version: "" # required - specify version of REMARK standard used
references: # Formatted metadata of original paper, from BibTex
  - type: article
    authors: # required
      -
        family-names: "Author 1 Last Name"
        given-names: "Author 1 First Name"
        orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
      -
        family-names: "Author 2 Last Name"
        given-names: "Author 2 First Name"
        orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
    title: "Title of original paper" # required
    doi: "Original paper DOI" # optional
    date: 20XX-XX-XX
    publisher : "Publisher information"
repository: "URL of repository" # optional

# Econ-ARK website fields? 
github_repo_url: Link to git repo hosting the code # required 
remark-name: template # required
title-original-paper: Name of the paper if available # optional 

dashboards: # path to any dashboards within the repo - optional
  - 
    path_to_dashboard.ipynb
identifiers-paper: # required for Replications; optional for Reproductions
   - 
      type: url 
      value: template
   - 
      type: doi
      value: doi:template
date-published-original-paper: 2020-09-14 # required for Replications; optional for Reproductions

tags: # Use the relavent tags
  - REMARK
  - Notebook

identifiers: # optional
  - 
    type: url
    value: "URL to project page"
  - 
    type: doi
    value: "DOI link if available"
keywords: # optional
  - Econ-ARK
  - Sample
  - Template

---

# Template metadata for REMARKs

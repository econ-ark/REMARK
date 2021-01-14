---
tags: # Use the relavent tags
  - REMARK
  - Notebook
abstract: "The abstract is optional" # abstract: optional
authors: # required
  -
    family-names: "Author 1 Last Name"
    given-names: "Author 1 First Name"
    orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
  -
    family-names: "Author 2 Last Name"
    given-names: "Author 2 First Name"
    orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
    
cff-version: "1.1.0" # required 
date-released: 20XX-XX-XX # required
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
message: "Description about the project" # required
repository-code: "Link to publicly available code" # optional
title: "Template for REMARK metadata" # required
version: "1.0" # required
# REMARK fields
github_repo_url: Link to git repo hosting the code # required 
commit: # Git commit number that the REMARK will always use; required for "frozen" remarks, optional for "draft" remarks
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
---

# Template metadata for REMARKs

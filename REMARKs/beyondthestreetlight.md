---
# CFF required fields
cff-version: 1.1.0 # required (don't change)
message: If you use this software, please cite it as below. # optional
authors: # required
  - family-names: Edwards
    given-names: Decory
    #orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
  - family-names: Carroll
    given-names: Christopher
    orcid: https://orcid.org/0000-0003-3732-9312
title: Beyond the streetlight # required
abstract: This repository provides an analysis of the trend in forecast errors made by the Tealbook/Greenbook(GB) and the Survey of Professional Forecasters(SPF) for measures of the unemployment rate and real growth in personal consumption expenditures from 1982 to 2017. # optional
version: 1.0.4 # optional Version of the software released
date-released: 2023-11-06 # required

# REMARK required fields
remark-version: 1.0 # required - specify version of REMARK standard used
references: # required for replications; optional for reproductions; BibTex data from original paper
  - type: article
    authors: # required
      - family-names: "Corrado"
        given-names: "Carol"
        #orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
      - family-names: "Kennickell"
        given-names: "Arthur"
        #orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
    title: "100 years of Economic Measurement in the Division of Research & Statistics: Beyond the streetlight" # required
    #doi: "Original paper DOI" # optional
    date: 2023-10-24 # required
    publisher: "R&S Centennial Conference"
#repository: "URL of repository" # optional (when original paper has own repository)

# Econ-ARK website fields 
github_repo_url: https://github.com/dedwar65/beyond-the-streetlight # required 
remark-name: beyond-the-streetlight # required
title-original-paper: "100 years of Economic Measurement in the Division of Research & Statistics: Beyond the streetlight" # optional 
notebooks:  # path to any notebooks within the repo - optional
  - RS100_Discussion_Slides.ipynb

tags: # Use the relavent tags
  - REMARK
  - Notebook

keywords: # optional
  - Econ-ARK
  - Greenbook
  - Forecast errors

---

# Abstract

This repository provides an analysis of the trend in forecast errors made by the Tealbook/Greenbook(GB) and the Survey of Professional Forecasters(SPF) for measures of the unemployment rate and real growth in personal consumption expenditures from 1982 to 2017. The data on forecasts for unemployment and consumption made by the federal reserve (Tealbook/Greenbook) and the mean across private forcasters are provided by the Philidelphia Fed. Data on realized values of the forecasted variables are provided by the St. Louis Fed. 

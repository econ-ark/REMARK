---
# CFF required fields
cff-version: 1.2.0
title: "Deep learning for solving dynamic economic models"
message: >-
  If you use this software, please cite it using the
  metadata from this file.
type: software
authors:
  - given-names: Lilia
    family-names: Maliar
    affiliation: >-
      a The Graduate Center, City University of New York,
      CEPR, and Hoover Institution, Stanford University
  - given-names: Serguei
    family-names: Maliar
    affiliation: Santa Clara University
  - given-names: Pablo
    family-names: Winant
    affiliation: ESCP Business School and CREST/Ecole Polytechnique
identifiers:
  - type: doi
    value: 10.1016/j.jmoneco.2021.07.004
abstract: >-
  We introduce a unified deep learning method that solves
  dynamic economic models by casting them into nonlinear
  regression equations. We derive such equations for three
  fundamental objects of economic dynamics – lifetime reward
  functions, Bellman equations and Euler equations. We
  estimate the decision functions on simulated data using a
  stochastic gradient descent method. We introduce an
  all-in-one integration operator that facilitates
  approximation of high-dimensional integrals. We use neural
  networks to perform model reduction and to handle
  multicollinearity. Our deep learning method is tractable
  in large-scale problems, e.g., Krusell and Smith (1998).
  We provide a TensorFlow code that accommodates a variety
  of applications.
keywords:
  - Artificial intelligence
  - Machine learning
  - Deep learning
  - Neural network
  - Stochastic gradient
  - Dynamic models
  - Model reduction
  - Dynamic programming
  - Bellman equation
  - Euler equation
  - Value function
references:
  - type: article
    authors:
      - family-names: "Krusell"
        given-names: "Per"
      - family-names: "Smith, Jr."
        given-names: "Anthony A."
    title: "Income and Wealth Heterogeneity in the Macroeconomy"
    doi: "10.1086/250034"
    date-released: 1998-10-01
    publisher:
        name: "Journal of Political Economy"

# REMARK fields
remark-version: "v1.0.0"
remark-name: "DeepLearningKrusselSmith"
github_repo_url: https://github.com/marcmaliar/deep-learning-euler-method-krusell-smith/
notebooks:
  - code/python/Main_KS.ipynb
---

# Deep learning for solving dynamic economic models

This notebook solves a version of Krusell and Smith's (1998) heterogenous-agent model with idiosyncrastic and aggregate shocks, incomplete markets and borrowing constraints. It uses a deep learning Euler-equation method introduced by Maliar, Maliar and Winant (2018) in the paper "Deep learning for solving dynamic economic models", Journal of Monetary Economics 122, pp 76-101. https://lmaliar.ws.gc.cuny.edu/files/2021/09/JME2021.pdf

We show a version of the Euler equation method that minimizes the sum of squared residuals in the equilibrium conditions. See [https://deepecon.org](https://deepecon.org) for documentation, updates and the other versions of the deep learning method (Bellman equation and life-time reward).

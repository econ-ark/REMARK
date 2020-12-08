# R[eplications/eproductions] and Explorations Made using ARK (REMARK)

This collects and organizes self-contained and complete projects written using the Econ-ARK.
The content here should be executable by anyone with a suitably configured computer (see "Installation.md"
in this directory).

Each project lives in its own repository. To make a new REMARK, please see the [REMARK starter example](https://github.com/econ-ark/REMARK-starter-example).  

Types of content include (see below for elaboration):

1. Explorations
   * Use the Econ-ARK/HARK toolkit to demonstrate some set of modeling ideas
1. Replications
   * Attempts to replicate important results of published papers written using other tools
1. Reproductions
   * Code that reproduces ALL of the results of some paper that was originally written using the toolkit

## REMARKs

| | REMARK       |  Link to REMARK |
| --| ------------ | ----------------|
| 0. | REMARK Template | https://github.com/econ-ark/REMARK-template |
| 1. | Public Debt and Low Interest Rates [Replication of Blanchard (2019)]            | https://github.com/econ-ark/BlanchardPA2019                |
| 2. | Solving heterogeneous agent models in discrete time with many idiosyncratic states by perturbation methods | https://github.com/econ-ark/BayerLuetticke |
| 3. | Theoretical Foundations of Buffer Stock Saving | https://github.com/econ-ark/BufferStockTheory |
| 4. | Consumption and Portfolio Choice Over the Life Cycle | https://github.com/econ-ark/CGMPortfolio |
| 5. | Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis | https://github.com/econ-ark/Carroll_1997_QJE |
| 6. | Consumer Spending During Unemployment: Positive and Normative Implications | |
| 7. | Income and wealth heterogeneity in the macroeconomy (KrusellSmith) | https://github.com/econ-ark/KrusellSmith |
| 8. | Liquidity Constraints and Precautionary Saving | https://github.com/llorracc/LiqConstr |
| 9. | Modeling the Consumption Response to the CARES Act | https://github.com/econ-ark/Pandemic |
| 10. | Optimal Financial Investment over the Life Cycle - Blog Post | https://github.com/econ-ark/PortfolioChoiceBlogPost |
| 11. | SolvingMicroDSOPs | |
| 12. | Sticky Expectations and Consumption Dynamics | https://github.com/llorracc/cAndCwithStickyE |
| 13. | Can Persistent Unobserved Heterogeneity in Returns-to-Wealth Explain Wealth Inequality? | https://github.com/econ-ark/cstwMPC-RHetero  |
| 14. | The Distribution of Wealth and the Marginal Propensity to Consume | https://github.com/llorracc/cstwMPC |
| 15. | Analytically tractable model of the effects of nonfinancial risk on intertemporal choice | https://github.com/llorracc/ctDiscrete |
| 16. | Endogenous Retirement: A Canonical Discrete-Continuous Problem | https://github.com/econ-ark/EndogenousRetirement |

## REMARK Guidelines

The only absolute requirements are:

1. The necessary information to strictly and robustly indicate the versions of ALL software required to recreate the environment in which the code works
  - For python projects, this consists of a `reproduce.sh` file in the binder directory
a reproduce.sh script that
     - Checks whether the required environment exists, and if not installs it
     - Runs and reproduces all the results
     
2. Strongly recommended are:
  - If reproduce.sh takes longer than a few minutes, a `reproduce_min.sh` that generates some interesting subset of results within a few minutes
  - A Jupyter notebook that exposits the material being reproduced

3. For a maximalist REMARK (the extra stuff is completely optional) included:
  - A reproduce_text-etc.sh that generates the text
  - A dashboard that creates interactive versions of interesting figures

## Replications and Reproductions

<!--
The [ballpark](http://github.com/econ-ark/ballpark) is a place for the set of papers that we would be delighted to have replicated in the Econ-ARK.

This REMARK repo is where we intend to store such replications (as well as the code for papers whose codebase was originally written using the Econ-ARK).
-->

In cases where the replication's author is satisfied that the main results of the paper have been successfully replicated, we expect to approve pull requests with minimal review.

We also expect to approve with little review cases where the author has a clear explanation of discrepancies between the paper's published results and the results in the replication attempt.

We are NOT intending this resource to be viewed as an endorsement of the replication; instead, it is a place for it to be posted publicly for other people to see and form judgments on. Nevertheless, pull requests for attempted replications that are unsuccessful for unknown reasons will require a bit more attention from the Econ-ARK staff, which may include contacting the original author(s) to see if they can explain the discrepancies, or may include consulting with experts in the particular area in question.

Replication archives should contain two kinds of content (along with explanatory material):
Code that attempts to comprehensively replicate the results of the paper, and a Jupyter notebook that presents at least a minimal set of examples of the use of the code.

This material will all be stored in a directory with some short pithy name (a bibtex citekey might make a good directory name) which, if written in an Econ-ARK compatible style, will also be the name of a module that other users can import and use.

Code archives should contain:
   * All information required to get the replication code to run
   * An indication of how long that takes on some particular machine

Jupyter notebook(s) should:
   * Explain their own content ("This notebook uses the associated replication archive to demonstrate three central results from the paper of [original author]: The consumption function and the distribution of wealth)
   * Be usable for someone wanting to explore the replication interactively (so, no cell should take more than a minute or two to execute on a laptop)

## Differences with DemARK

The key difference with the contents of the [DemARK](https://github.com/econ-ark/DemARK) repo is that REMARKs are allowed to rely on the existence of local files and subdirectories (figures; data) at a predictable filepath relative to the location of the root.

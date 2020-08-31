# R[eplications/eproductions] and Explorations Made using ARK (REMARK)

This collects and organizes self-contained and complete projects written using the Econ-ARK.
The content here should be executable by anyone with a suitably configured computer (see "Installation.md"
in this directory).

Each project lives in its own subdirectory in the `REMARKs` directory.  

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
| 3. | Theoretical Foundations of Buffer Stock Saving | https://github.com/llorracc/BufferStockTheory |
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

Every remark *must* have the following content (described in more detail in sections below):
1. `do_all_code.sh`
    * This should be at the root level of the directory
1. In `Code/Python` an eponymous Jupyter Notebook
    * For example, `Code/Python/BufferStockTheory.ipynb`
1. In the REMARKs directory, an eponymous markdown file with a brief description
    * For example, `REMARKs/BufferStockTheory.md`
1. In the REMARKs directory, an eponymous bibtex file containing a reference
    * For example, `REMARKs/BufferStockTheory.bib`

It is *highly encouraged* for each REMARK to adhere to the following guidelines:

1. The Jupyter notebook should be a `.py` file bound to the notebook with Jupytext.
1. `do_all.py` should execute this notebook code
1. Executing `do_all.py` should save figures to a `Figures/` directory.

### `do_[].py`

Each REMARK directory should contain a file or files beginning with the word `do` and ending with a `.py` extension. The only such file that is absolutely required is `do_all.py.` If executing everything in the code takes more than a few minutes, there should also be a `do_min.py.` Other files that are intermediate between `do_min` and `do_all` are optional.

* `do_all.py` should produce all of the results that the tool is capable of generating
   * In other words, `do_all.py` should execute all the economic simulation code.
   * For a reproduction, it should produce substantially all of the tables and figures of the associated paper
   * For a replication, it should produce whatever the author judges to be a characteristic set of the results of the paper
   * When executed, it should:
      * Inform the user of the minimal resources required for execution (RAM, processor speed, etc)
	  * Tell the user roughly how long execution takes on a machine that satisfies those requirements
	  * Get the user's permission before proceeding

### Eponymous Jupyter Notebook

For a project named `ThisIsMyREMARK` there should be a Jupyter notebook named `ThisIsMyREMARK.ipynb`  in the directory `Code/Python`

The preferred, but not required, practice is for the Jupyter notebook to be the generator of the python code executed by `do_all.py.`  An example of how this can be accomplished is given by the `BufferStockTheory`
REMARK

### An Eponymous Markdown file

This should contain a brief description of the project, which will be used on the website as the brief description of the REMARK

This is an unstructured category, whose purpose to hold pretty much any kind of self-contained and coherent exercise. Purposes might include:

1. Illustrations of the uses of a particular model
1. Examples of how to use a particular technique (e.g., indirect inference)
1. Comparisons of the results of different models to each other

and pretty much anything else that uses the toolkit but does not fall into the category of replications or reproductions of a paper

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

# R[eplications/eproductions] and Explorations Made using ARK (REMARK)

REMARKs are self-contained and complete projects, whose content here should be executable by anyone with a suitably configured computer or using [nbreproduce](https://econ-ark.github.io/nbreproduce/).

Types of content include (see below for elaboration):

1. Explorations
   * Use the Econ-ARK/HARK toolkit to demonstrate some set of modeling ideas
1. Replications
   * Attempts to replicate important results of published papers written using other tools
1. Reproductions
   * Code that reproduces ALL of the results of some paper that was originally written using the toolkit

## For Authors

Each project lives in its own repository. To make a new REMARK, you can start with a skeleton
[REMARK-starter-example](https://github.com/econ-ark/REMARK-starter-example) and add to it,
or from an example of a complete project using the toolkit, [BufferStockTheory](https://github.com/econ-ark/BufferStockTheory), whose content (code, text, figures, etc) you can replace with
your own.

REMARKs should adhere to the [REMARK Standard](https://github.com/econ-ark/REMARK/blob/master/STANDARD.md).

## For Editors

The REMARK catalog and Econ-ARK website configuration will be maintained by Editors.

Editorial guidelines are [here](https://github.com/econ-ark/REMARK/blob/master/EDITORIAL.md).

## REMARK Catalog

A catalog of all REMARKs  is available under the `REMARK` tab at [econ-ark.org](https://econ-ark.org/materials).

The [ballpark](http://github.com/econ-ark/ballpark) is a place for the set of papers that we would be delighted to have replicated in the Econ-ARK.

In cases where the replication's author is satisfied that the main results of the paper have been successfully replicated, we expect to approve pull requests for new REMARKs with minimal review, as long as they satsify the criteria described in the [Standard](https://github.com/econ-ark/REMARK/blob/master/STANDARD.md).

We also expect to approve with little review cases where the author has a clear explanation of discrepancies between the paper's published results and the results in the replication attempt.

We are NOT intending this resource to be viewed as an endorsement of the replication; instead, it is a place for itb to be posted publicly for other people to see and form judgments on. Nevertheless, pull requests for attempted replications that are unsuccessful for unknown reasons will require a bit more attention from the Econ-ARK staff, which may include contacting the original author(s) to see if they can explain the discrepancies, or may include consulting with experts in the particular area in question.

Replication archives should contain two kinds of content (along with explanatory material):

1. Code that attempts to replicate key results of the paper
1. A Jupyter notebook that presents at least a minimal set of examples of the use of the code.

This material will all be stored in a directory with some short pithy name (a bibtex citekey might make a good directory name).

Code archives should contain:
   * All information required to get the replication code to run
      * Including a `requirements.txt` file explaining the software requirements
   * An indication of how long it takes to run the `reproduce.sh` script
      * One some particular machine whose characteristics should be described

Jupyter notebook(s) should:
   * Explain their own content ("This notebook uses the associated replication archive to demonstrate three central results from the paper of [original author]: The consumption function and the distribution of wealth)
   * Be usable for someone wanting to explore the replication interactively (so, no cell should take more than a minute or two to execute on a laptop)

## Differences with DemARK

The key difference with the contents of the [DemARK](https://github.com/econ-ark/DemARK) repo is that REMARKs are allowed to rely on the existence of local files and subdirectories (figures; data) at a predictable filepath relative to the location of the root.

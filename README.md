# R[eplications/eproductions] and Explorations Made using ARK (REMARK)

The Economics profession has fallen behind other fields in the adoption of modern software development and reproducibility practices.
Economists generally lack robust software engineering training and resort to writing new software from scratch.
This practice of 'reinventing the wheel' increases the likelihood of errors and undermines the reproducibility of computational results.

Poor software development practices also slow down the advancement of Economics as a science.
When researchers cannot reproduce the results of a paper, they are unable to build on the work of others, which leads to wasted time and effort and slows down progress.
Additionally, when researchers cannot reproduce the results of a paper, they are unable to verify the validity of said results.
This leads to a lack of trust in the research, which can have a negative impact on the influence of the field.

Other disciplines have adapted to modern software development methods, and it is time for Economics to catch up.

To address the problem discussed above, Econ-ARK has been working on developing a set of standards and tools for reproducibility in our own work in Economics.
Our reproducibility initiative is called REMARK, which stands for "R[eplications/eproductions] and Explorations Made using ARK".
The term REMARK is used to represent the standard, as well as any project that follows the standard.
The objective of REMARKs is to be self-contained and complete projects, whose contents should be executable by anyone on any computer that meets a minimal set of requirements and software.

The REMARK standard is focused on 3 key principles:

- **Reproduction**: The ability to reproduce the results of a project using the same data and code on a different computer.
- **Archiving**: Storing the project in a way that it can be accessed and used in the future.
- **Publishing**: Making the project available to the public and incentivizing the sharing of code and data.

Detailed technical instructions for using or contributing to REMARK can be found in UsingREMARK.md in this directory.

## Description

REMARKs are self-contained and complete projects, whose content here should be executable by anyone with a suitably configured computer or using [nbreproduce](https://econ-ark.github.io/nbreproduce/).

Types of content include (see below for elaboration):

1. Explorations
   * Use the Econ-ARK/HARK toolkit to demonstrate some set of modeling ideas
1. Replications
   * Attempts to replicate important results of published papers written using other tools
1. Reproductions
   * Code that reproduces ALL of the results of some paper that was originally written using the toolkit

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


# REMARK: Replications and Explorations Made using the ARK

This is the resting place for self-contained and complete projects written using the Econ-ARK.
The content here should be executable by anyone with a suitably configured computer (see "Installation.md"
in this directory).

Each project lives in its own subdirectory in the `REMARKs` directory

Types of content include (see below for elaboration):

1. Explorations
   * Use the Econ-ARK/HARK toolkit to demonstrate some set of modeling ideas 
1. Replications
   * Attempts to replicate the results of published papers written using other tools
1. Reproductions
   * Code that reproduces the results of some paper that was originally written using the toolkit

## `do_[].py`

A common feature of all of the content here is that the root level of each directory should contain a file or files beginning with the word `do` and ending with a `.py` extension. The only such file that is absolutely required is `do_all.py.` If executing everything in the code takes more than a few minutes, there should also be a `do_min.py.` Other files that are intermediate between `do_min` and `do_all` are optional.

* `do_min.py` should produce the minimal set of results that might be useful
   * The most important definition of _minimal_ is that `do_min.py` files should execute in at most a minute or two on a mid-range laptop computer
   * Given that constraint, the `do_min.py` file should illustrate as much as possible of the content of the codebase
* `do_mid.py` should produce a representative range of the most interesting and important results
   * When executed, it should:
      * Inform the user of the minimal resources required for execution (RAM, processor speed, etc)
	  * Tell the user roughly how long execution takes on a machine that satisfies those requirements
	  * Get the user's permission before proceeding 
* `do_all.py` should produce all of the results that the tool is capable of generating
   * For example, ideally for a reproduction or replication, it should produce all of the tables and figures of the associated paper 
   * When executed, it should:
      * Inform the user of the minimal resources required for execution (RAM, processor speed, etc)
	  * Tell the user roughly how long execution takes on a machine that satisfies those requirements
	  * Get the user's permission before proceeding 

# Explorations

This is an unstructured category, designed to hold pretty much any kind of self-contained and coherent exercise. Purposes might include:

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
   


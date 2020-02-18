# Cocco, Gomes, & Maenhout (2005), "[Consumption and Portfolio Choice Over the Life Cycle](https://academic.oup.com/rfs/article-abstract/18/2/491/1599892)"

# A replication by [Mateo Velásquez-Giraldo](https://mv77.github.io/) and [Matthew Zahn](https://sites.google.com/view/matthew-v-zahn/matthew-v-zahn).

**Quick launch**: the following link launches a Jupyter notebook with the main results.
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/matthew-zahn/CGMPort/master?filepath=CGMPort-Public%2FCode%2FPython%2FCGM_REMARK.ipynb) **TODO: Point link to REMARK repo after merge. Currently points to MVZ's repo.**

## Description

This repository corresponds to the [Final assignment](https://github.com/ccarrollATjhuecon/Methods/blob/master/Assignments/14_Final-Class-Project/Final-Class-Project.md) of the course Advanced Macroeconomics I at Johns Hopkins University, by Professor Christopher D. Carroll.

## Location of main files:
1. A notebook attempting to replicate the paper's main results using HARK can be found at <tt>Code/Python/CGMPortfolio.ipynb</tt>.

2. A document going into more detail on our attempt to replicate can be found at <tt>CGMPortfolio.pdf</tt>.

3. The code that generates all the figures and results in the previous document can be found at <tt>Code/Python</tt>. Files can be run independently, or all at once through the script <tt>./do_ALL.py</tt>. Additional files <tt>./do_MIN.py</tt> and <tt>./do_MID.py</tt> are made available to execute subsets of the results.
- <tt>./do_MIN.py</tt>: solves the baseline model plotting its policy functions, and presents mean simulated life-cycle behavior of variables of interest. Runtime: ~400 seconds.
- <tt>./do_MID.py</tt>: additionally compares the policy functions obtained with HARK with those that we obtain from executing the authors' FORTRAN 90 code. Runtime: ~600 seconds.
- <tt>./do_ALL.py</tt>: additionally computes all the results from the apendices, in which we alter the baseline model in HARK to cases in which analytical solutions are available. Runtime: ~1300 seconds.

Note: runtimes are estimated using an Intel Core i7-6700HQ CPU @ 2.60GHz.

4. The original Fortran 90 code made available by the authors can be found in <tt>Code/Fortran</tt>.

## Package requirements

The following packages are required for executing the <tt>REMARK</tt> notebook:
- matplotlib
- numpy
- scipy
- pandas
- econ-ark >= 0.10.3

For executing <tt>do_ALL.py</tt> additional requirements are:
- seaborn

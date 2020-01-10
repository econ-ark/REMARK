# Cocco, Gomes, & Maenhout (2005), "[Consumption and Portfolio Choice Over the Life Cycle](https://academic.oup.com/rfs/article-abstract/18/2/491/1599892)"

# A replication by [Mateo Velásquez-Giraldo](https://github.com/Mv77) and [Matthew Zahn](https://sites.google.com/view/matthew-v-zahn/matthew-v-zahn).

**Quick launch**: the following link launches a Jupyter notebook with the main results.
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/matthew-zahn/CGMPort/master?filepath=CGMPort-Public%2FCode%2FPython%2FCGM_REMARK.ipynb) **TODO: Point link to REMARK repo after merge. Currently points to MVZ's repo.**

## Description

This repository corresponds to the [Final assignment](https://github.com/ccarrollATjhuecon/Methods/blob/master/Assignments/14_Final-Class-Project/Final-Class-Project.md) of the course Advanced Macroeconomics I at Johns Hopkins University, by Professor Christopher D. Carroll.

## Location of main files:
1. A notebook attempting to replicate the paper's main results using HARK can be found at <tt>Code/Python/CGMPortfolio.ipynb</tt>.

1. A document going into more detail on our attempt to replicate can be found at <tt>CGMPortfolio.pdf</tt>.

1. The code that generates all the figures and results in the previous document can be found at <tt>Code/Python</tt>. Files can be run independently, or all at once through the script <tt>./do_ALL.py</tt>.

1. The original Fortran 90 code made available by the authors can be found in <tt>Code/Fortran</tt>.

## Package requirements

The following packages are required for executing the <tt>REMARK</tt> notebook:
- matplotlib
- numpy
- scipy
- pandas
- econ-ark >= 0.10.3

For executing <tt>do_ALL.py</tt> additional requirements are:
- seaborn

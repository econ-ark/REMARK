# GM2003
This is a REMARK of [Gomes and Michaelides (2003)](https://www.sciencedirect.com/science/article/abs/pii/S1094202503000590), Portfolio choice with internal habit formation:
a life-cycle model with uninsurable labor
income risk. This file is modified based on CGMPortfolio.ipynb by Mateo Velásquez-Giraldo and Matthew Zahn that intended to replicate main results in Cocco et al. (2005).
The file makes some comparison and contrast to the main results of two papers.
## New HARK consumer type development required(unfinished)
Some key resuls in this paper were not replicated. To replicate them may require to build a new HARK consumer type by changing the solution method of the agent since now we have
two state variables in consumer's value function. In addtion, the addition of habits change the utility function to time inseperable. A simple step is to add habit to the simplest consumer type, PerfForesightConsumerType, and also modify the coreesponding solution method. The current unfinished new consumer type, ConsHabitModel, and, ConsPortfolioHabitModel_with_Hgamma=1, can be found in the directory UnfinishedNewCosumerType.
The construction is now stucked at the critical process to build habit grid and transition matrix, according to appendix B in this paper, and section 7 in [Carroll (1999)](https://library.wolfram.com/infocenter/MathSource/832/SolvingMicroDSOP.pdf) explaining numerically solving question with multiple state variables. (the newer version of this lecture note does not include this section). It may need further suggestions from HARK experts to see how to realize this using HARK tools.

## python environment

The do_all_code.sh file will generate everything. The script took around 193.14052391052246 second to run with 8.00 GB RAM and Intel Core i7-7500U CPU @ 2.70GHz 2.90 GHz

Figures and tables will be generated with the do_all.py file
The do_all.py file will run generate all of the relevant figures and tables and save them in their respective directories.

Figures will be saved in /Figures
The do_all.py file is similar to the GM2003.ipynb notebook. They are different in that the do_all.py file includes code that saves the figures.
The notebook is located in /Code/Python. It is named "GM2003.ipynb".

The Econ-ARK/HARK toolkit must be preinstalled. Please refer to https://github.com/econ-ark/HARK to install the required toolkit.

Package requirements
The following packages are required for executing the REMARK notebook:

matplotlib
numpy
scipy
pandas
econ-ark >= 0.10.3

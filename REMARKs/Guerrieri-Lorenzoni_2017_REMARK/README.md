# Guerrieri & Lorenzoni (2017), "[Credit Crises, Liquidity Constraints, and Precautionary Savings]"

# A replication by [William Du] and [Tung-Sheng-Hsieh].

## Description

This repository corresponds to the [Final assignment](https://github.com/ccarrollATjhuecon/Methods/blob/master/Assignments/14_Final-Class-Project/Final-Class-Project.md) of the course Computational Methods at Johns Hopkins University, by Professor Christopher D. Carroll.

## Location of main files:
  1. A notebook attempting to replicate the paper's main results using HARK can be found at Code/Python/Guerrieri-Lorenzoni.ipynb.

  2. A document going into more detail on our attempt to replicate can be found at GL2017.pdf.

  3. The code that generates all the figures and results in the previous document can be found at Code/Python. Files can be run independently, 
     or all at once through the scrip ./do_ALL.py. 

  4. ./do_ALL.py: additionally computes all the results.  Runtime: ~8 seconds.

  5. The original MATLAB code made available by the authors can be found in Code/MATLAB
  
  6. The python code requires inc_process.mat and cl.mat to run

## Package/data requirements

The following packages and data files are required for executing the <tt>REMARK</tt> notebook:
- numpy
- scipy
- econ-ark >= 0.10.8
- cl.mat and inc_process.mat files

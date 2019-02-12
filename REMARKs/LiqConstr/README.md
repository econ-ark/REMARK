# Reproducing the Results

The results (figures) of this paper are generated using the [Econ-ARK](https://econ-ark.org) toolkit. 

In order to run the `do_all.py` file to ['reproduce'](https://github.com/econ-ark/REMARK)  the paper's results, you 
will need python version 3 installed on your computer.

In addition, you will need to have installed the required packages listed in the requirements.txt file. Assuming that you have the 'pip' tool on your computer's command line, you can ensure the necessary packages are installed using:

	pip install -r requirements.txt

Once these requirements are satisfied, you should be able to generate the paper's figures from the command line via 

	ipython do_all.py
	
or you should be able to execute the interactive jupyter notebook `LiqConstr.ipynb` from a suitably configured computer.  (Installation of the [anaconda](https://anaconda.org) superset of python will install the jupyter notebook command that should allow you to execute the notebook).



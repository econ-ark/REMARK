This is a REMARK of Carroll (1997), Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis.

The doEverything.sh file will generate everything. It will take around 1~2 minutes for everything to run.
* Figures and tables will be generated with the do_all.py file
* The main paper is generated in /Paper
* Slides are generated in /Slides

The do_all.py file will run generate all of the relevant figures and tables and save them in their respective directories.
* Figures will be saved in /Paper/Figures
* Tables will be saved in /Paper/Tables
* The do_all.py file is similar to the Carroll_1997_QJE.ipynb notebook. They are different in that the do_all.py file includes code that saves the figures and tables.

The notebook is located in /Code/Python. It is named "Carroll_1997_QJE.ipynb".
* The notebook creates a clone of itself as a .py file if it is run.

The Econ-ARK/HARK toolkit must be preinstalled. Please refer to https://github.com/econ-ark/HARK to install the required toolkit.

Tables are generated with "tabulate". If you do not have tabulate preinstalled, install tabulate with pip by running the following code in the terminal:
* pip install tabulate
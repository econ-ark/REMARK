# %% {"hidden": true}
# This cell does some setup; please be patient, it may take 3-5 minutes

# The tools for navigating the filesystem
import sys
import os

# This is a jupytext paired notebook that autogenerates BufferStockTheory.py
# which can be executed from a terminal command line via "ipython BufferStockTheory.py"
# But a terminal does not permit inline figures, so we need to test jupyter vs terminal
# Google "how can I check if code is executed in the ipython notebook"

from IPython import get_ipython # In case it was run from python instead of ipython

# If the ipython process contains 'terminal' assume not in a notebook
def in_ipynb():
    try:
        if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
            return True
        else:
            return False
    except NameError:
        return False
    
# Import related generic python packages
import numpy as np
from time import clock
mystr = lambda number : "{:.4f}".format(number)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show

# In order to use LaTeX to manage all text layout in our figures, 
# we import rc settings from matplotlib.
from matplotlib import rc

# The warnings package allows us to ignore some harmless but alarming warning messages
import warnings
warnings.filterwarnings("ignore")

from copy import copy, deepcopy

# Code to allow a master "Generator" and derived "Generated" versions
Generator=False # Is this notebook the master or is it generated?

# Determine whether to make the figures inline (for spyder or jupyter)
# vs whatever is the automatic setting that will apply if run from the terminal
if in_ipynb():
    # %matplotlib inline generates a syntax error when run from the shell
    # so do this instead
    get_ipython().run_line_magic('matplotlib', 'inline')
else:
    get_ipython().run_line_magic('matplotlib', 'auto')
    Generator = True

# Define (and create, if necessary) the figures directory "Figures"
if Generator:
    my_file_path = os.path.dirname(os.path.abspath("BufferStockTheory.ipynb")) # Find pathname to this file:
    Figures_HARK_dir = os.path.join(my_file_path,"Figures/") # LaTeX document assumes figures will be here
    Figures_HARK_dir = os.path.join(my_file_path,"/tmp/Figures/") # Uncomment to make figures outside of git path
    if not os.path.exists(Figures_HARK_dir):
        os.makedirs(Figures_HARK_dir)
        
if not in_ipynb(): # running in batch mode
    print('You appear to be running from a terminal')
    print('By default, figures will appear one by one')

def show(target_dir,figure_name):
    # Save the figures in several formats
    if Generator:
        print(f"Saving figure {figure_name} in {target_dir}")
        plt.savefig(os.path.join(target_dir, f'{figure_name}.png'))
        #  plt.savefig(os.path.join(target_dir, f'{figure_name}.jpg'))
        #  plt.savefig(os.path.join(target_dir, f'{figure_name}.pdf'))
        #  plt.savefig(os.path.join(target_dir, f'{figure_name}.svg'))
        plt.clf()
    #if not in_ipynb():
    #    plt.ioff()
    #    plt.draw()
    #    #    plt.show(block=False) 
    #    plt.pause(1)
    else:
        plt.show(block=True) # Change to False if you want to run uninterrupted

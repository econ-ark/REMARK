import sys
import os

from IPython import get_ipython # In case it was run from python instead of ipython

# If the ipython process contains 'terminal' assume not in a notebook
def in_ipynb():
    try:
        if 'terminal' in str(type(get_ipython())):
            return False
        else:
            return True
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

# 20191113 CDC to Seb: I do not yet have a KrusellSmith-Problems-And-Solutions.ipynb file
# If I DID, that notebook would be the "Generator" and it would generate KrusellSmith-Problems.ipynb and KrusellSmith.ipynb
# So, once you have processed this, delete this block of comments
# Code to allow a master "Generator" and derived "Generated" versions
#   - allows "$nb-Problems-And-Solutions → $nb-Problems → $nb"
# Generator=False # Is this notebook the master or is it generated?

# Determine whether to make the figures inline (for spyder or jupyter)
# vs whatever is the automatic setting that will apply if run from the terminal
if in_ipynb():
    # %matplotlib inline generates a syntax error when run from the shell
    # so do this instead
    get_ipython().run_line_magic('matplotlib', 'inline')
else:
    get_ipython().run_line_magic('matplotlib', 'auto')
    showFigs = True

# Whether to save the figures in a local directory
saveFigs=True
# Whether to draw the figs on the GUI (if there is one)
drawFigs=True # 20191113 CDC to Seb: Is there a way to determine whether we are running in an environment capable of displaying graphical figures?  @mridul might know.  If there is, then we should have a line like the one below:
# if not inGUI:
#    drawFigs=False


# Define (and create, if necessary) the figures directory "Figures"
if saveFigs:
    my_file_path = os.path.dirname(os.path.abspath("BufferStockTheory.ipynb")) # Find pathname to this file:
    Figures_dir = os.path.join(my_file_path,"Figures/") # LaTeX document assumes figures will be here
    if not os.path.exists(Figures_dir):
        os.makedirs(Figures_dir)
        
if not in_ipynb(): # running in batch mode
    print('You appear to be running from a terminal')
    print('By default, figures will appear one by one')

def show(figure_name, target_dir="Figures"):
    # Save the figures in several formats
    if saveFigs:
        # print(f"Saving figure {figure_name} in {target_dir}") # Printing this clutters the interactive display; print to log, not terminal
        plt.savefig(os.path.join(target_dir, f'{figure_name}.png')) # For html4
        plt.savefig(os.path.join(target_dir, f'{figure_name}.jpg')) # For MSWord
        plt.savefig(os.path.join(target_dir, f'{figure_name}.pdf')) # For LaTeX
        plt.savefig(os.path.join(target_dir, f'{figure_name}.svg')) # For html5
        # plt.clf() # We do not want to close it because we haven't yet displayed it; delete this line
    if not in_ipynb():
        if drawFigs:
            plt.ioff()   # When plotting in the terminal, do not use interactive mode
            plt.draw()  
            plt.pause(2) # Wait a couple of secs to allow the figure to be briefly visible after being drawn
    else: # Running in Spyder or Jupyter or Jupyter Lab so OK to wait on user before continuing
        plt.show(block=True) # Change to False if you want to run uninterrupted

    plt.clf()
 

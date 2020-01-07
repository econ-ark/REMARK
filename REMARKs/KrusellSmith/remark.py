from IPython import get_ipython # In case it was run from python instead of ipython 
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

# The warnings package allows us to ignore some harmless but alarming warning messages
import warnings
warnings.filterwarnings("ignore")

# If the ipython process contains 'terminal' assume not in a notebook
# Case 1. In Notebook
def in_ipynb():
    try:
        if 'terminal' in str(type(get_ipython())):
            return False
        else:
            return True
    except NameError:
        return False

# Case 2. In IDE
def in_ide():
    ides = ['PYCHARM','SPYDER']

    if any([any([ide in name
            for ide
            in ides])
            for name
            in os.environ]):
        return True
   else:
        return False

## Case 3. Command line (possibly headless)
if not in_ipynb():
    print("Matplotlib backend: " + matplotlib.get_backend())

if not in_ipynb(): # running in batch mode
    print('You appear to be running from a terminal')
    print('By default, figures will appear one by one')


# Whether to draw the figs on the GUI (if there is one)
drawFigs=True # 20191113 CDC to Seb: Is there a way to determine whether we are running in an environment capable of displaying graphical figures?  @mridul might know.  If there is, then we should have a line like the one below:
# if not inGUI:
#    drawFigs=False

my_file_path = my_path = os.getcwd() # Path to this notebook
Figures_dir = os.path.join(my_file_path,"Figures/") # LaTeX document assumes figures will be here
if not os.path.exists(Figures_dir):
    os.makedirs(Figures_dir)
        

def show(figure_name, target_dir=Figures_dir):
    # Save the figures in several formats
    
    plt.savefig(os.path.join(target_dir, f'{figure_name}.png')) # For html4
    plt.savefig(os.path.join(target_dir, f'{figure_name}.jpg')) # For MSWord
    plt.savefig(os.path.join(target_dir, f'{figure_name}.pdf')) # For LaTeX
    plt.savefig(os.path.join(target_dir, f'{figure_name}.svg')) # For html5

    if not plt.isinteractive():
        plt.draw()

    plt.show()
    plt.clf()
 

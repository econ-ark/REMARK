# Runtime ~400 seconds
import sys
sys.path.append('./Code/Python/')

# %% Set up plot displays
import matplotlib
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto')

# %% Calibration assessment and life cycle simulations

# 1. Solve the model and display its policy functions
print('1. Solve the model and display its policy functions')
import Simulations.PolicyFuncs

# 2. Run a larger simulation to display the age conditional means of variables
# of interest.
print('2. Run a larger simulation to display the age conditional means of variables of interest.')
import Simulations.AgeMeans
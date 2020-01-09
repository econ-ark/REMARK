# Runtime ~600 seconds
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

# 2. Simulate the lives of a few agents to show the implied income
# and stockholding processes.
print('2. Simulate the lives of a few agents to show the implied income and stockholding processes.')
import Simulations.FewAgents

# 3. Run a larger simulation to display the age conditional means of variables
# of interest.
print('3. Run a larger simulation to display the age conditional means of variables of interest.')
import Simulations.AgeMeans

# %% Comparison

# 4. Solve and compare policy functions with those obtained from CGM's
# Fortran 90 code
print('4. Solve and compare policy functions with those obtained from CGM\'s Fortran 90 code')
import Comparison.ComparePolFuncs

# 5. Present more detailed figures on discrepancies for the last periods of
# life.
print('5. Present more detailed figures on discrepancies for the last periods of life.')
import Comparison.Compare_last_periods
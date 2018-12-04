from __future__ import division, print_function

from builtins import range
import numpy as np
import csv
from copy import  deepcopy
import os

# Set basic parameters 
working_T = 41*4              # Number of working periods
retired_T = 55*4              # Number of retired periods
T_cycle = working_T+retired_T # Total number of periods
CRRA = 1.0                    # Coefficient of relative risk aversion
DiscFac_guess = 0.99          # Initial starting point for discount factor
UnempPrb = 0.07               # Probability of unemployment while working
IncUnemp = 0.15               # Unemployment benefit replacement rate
BoroCnstArt = 0.0             # Artificial borrowing constraint
percentiles_to_match = [0.2, 0.4, 0.6, 0.8]    # Which points of the Lorenz curve to match in beta-dist (must be in (0,1))

# Set grid sizes
PermShkCount = 5              # Number of points in permanent income shock grid
TranShkCount = 5              # Number of points in transitory income shock grid
aXtraMin = 0.00001            # Minimum end-of-period assets in grid
aXtraMax = 20                 # Maximum end-of-period assets in grid
aXtraCount = 20               # Number of points in assets grid
aXtraNestFac = 3              # Number of times to 'exponentially nest' when constructing assets grid
CubicBool = False             # Whether to use cubic spline interpolation
vFuncBool = False             # Whether to calculate the value function during solution
do_agg_shocks = False         # Solve the FBS aggregate shocks version of the model

# Set simulation parameters
Population = 1000             # Total number of simulated agents in the population
T_sim_PY = 1200               # Number of periods to simulate (idiosyncratic shocks model, perpetual youth)
ignore_periods_PY = 400       # Number of periods to throw out when looking at history (perpetual youth)
T_age = T_cycle + 1           # Don't let simulated agents survive beyond this age
pLvlInitStd = 0.4             # Standard deviation of initial permanent income
aNrmInitMean = np.log(0.5)    # log initial wealth/income mean
aNrmInitStd  = 0.5            # log initial wealth/income standard deviation

# Set population macro parameters
PopGroFac = 1.01**(0.25)      # Population growth rate
PermGroFacAgg = 1.015**(0.25) # TFP growth rate

# Set indiividual parameters for the infinite horizon model
IndL = 10.0/9.0               # Labor supply per individual (constant)
PermGroFac_i = [1.000**0.25]  # Permanent income growth factor (no perm growth)
DiscFac_i = 0.97              # Default intertemporal discount factor
LivPrb_i = [1.0 - 1.0/160.0]  # Survival probability
PermShkStd_i = [(0.01*4/11)**0.5] # Standard deviation of permanent shocks to income
TranShkStd_i = [(0.01*4)**0.5]    # Standard deviation of transitory shocks to income

# Define the paths of permanent and transitory shocks (from Sabelhaus and Song)
TranShkStd = (np.concatenate((np.linspace(0.1,0.12,17), 0.12*np.ones(17), np.linspace(0.12,0.075,61), np.linspace(0.074,0.007,68), np.zeros(retired_T+1)))*4)**0.5
TranShkStd = np.ndarray.tolist(TranShkStd)
PermShkStd = np.concatenate((((0.00011342*(np.linspace(24,64.75,working_T-1)-47)**2 + 0.01)/(11.0/4.0))**0.5,np.zeros(retired_T+1)))
PermShkStd = np.ndarray.tolist(PermShkStd)

# Import the SCF wealth data
SCF_data_file = 'SCFwealthDataReduced.txt'
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '/' + SCF_data_file,'r')
SCF_reader = csv.reader(f,delimiter='\t')
SCF_raw = list(SCF_reader)
SCF_wealth = np.zeros(len(SCF_raw)) + np.nan
SCF_weights = deepcopy(SCF_wealth)
for j in range(len(SCF_raw)):
    SCF_wealth[j] = float(SCF_raw[j][0])
    SCF_weights[j] = float(SCF_raw[j][1])

# Make a dictionary for the infinite horizon type
init_infinite = {"CRRA":CRRA,
                "Rboro": 1.1248,           # Interest factor when borrowing
                "Rsave": 1.01/LivPrb_i[0],
                "PermGroFac":PermGroFac_i,
                "PermGroFacAgg":1.0,
                "BoroCnstArt":BoroCnstArt,
                "CubicBool":CubicBool,
                "vFuncBool":vFuncBool,
                "PermShkStd":PermShkStd_i,
                "PermShkCount":PermShkCount,
                "TranShkStd":TranShkStd_i,
                "TranShkCount":TranShkCount,
                "UnempPrb":UnempPrb,
                "IncUnemp":IncUnemp,
                "UnempPrbRet":None,
                "IncUnempRet":None,
                "aXtraMin":aXtraMin,
                "aXtraMax":aXtraMax,
                "aXtraCount":aXtraCount,
                "aXtraExtra":[None],
                "aXtraNestFac":aXtraNestFac,
                "LivPrb":LivPrb_i,
                "DiscFac":DiscFac_i, # dummy value, will be overwritten
                "cycles":0,
                "T_cycle":1,
                "T_retire":0,
                'T_sim':T_sim_PY,
                'T_age': 400,
                'IndL': IndL,
                'aNrmInitMean':np.log(0.00001),
                'aNrmInitStd':0.0,
                'pLvlInitMean':0.0,
                'pLvlInitStd':0.0,
                'AgentCount':0, # will be overwritten by parameter distributor
                }

# Make a base dictionary for the cstwMPCmarket
init_market = {'LorenzBool': False,
               'ManyStatsBool': False,
               'ignore_periods':0,    # Will get overwritten
               'PopGroFac':0.0,       # Will get overwritten
               'T_retire':0,          # Will get overwritten
               'TypeWeights':[],      # Will get overwritten
               'Population':Population,
               'act_T':0,             # Will get overwritten
               'IncUnemp':IncUnemp,
               'cutoffs':[(0.99,1),(0.9,1),(0.8,1),(0.6,0.8),(0.4,0.6),(0.2,0.4),(0.0,0.2)],
               'LorenzPercentiles':percentiles_to_match,
               'AggShockBool':do_agg_shocks
               }

def main():
    print("Sorry, SetupParamsCSTWnew doesn't actually do anything on its own.")
    print("This module is imported by cstwMPCnew, providing data and calibrated")
    print("parameters for the various estimations.  Please see that module if")
    print("you want more interesting output.")

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:10:36 2019

@author: mateo
"""

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import matplotlib.pyplot as plt
import pandas as pd

# %% Set up figure path
import sys,os

# Determine if this is being run as a standalone script
if __name__ == '__main__':
    # Running as a script
    my_file_path = os.path.abspath("../")
else:
    # Running from do_ALL
    my_file_path = os.path.dirname(os.path.abspath("do_ALL.py"))
    my_file_path = os.path.join(my_file_path,"Code/Python/")
    
FigPath = os.path.join(my_file_path,"Figures/")

# %% Calibration and solution
sys.path.append(my_file_path) 
# Loading the parameters from the ../Code/Calibration/params.py script
from Calibration.params import dict_portfolio, time_params

agent = cpm.PortfolioConsumerType(**dict_portfolio)
agent.solve()

# %% Run simulation and store results in a data frame

# Number of agents and periods in the simulation.
agent.AgentCount = 50 # Number of instances of the class to be simulated.
# Since agents can die, they are replaced by a new agent whenever they do.

# Number of periods to be simulated
agent.T_sim = agent.T_cycle*50

# Set up the variables we want to keep track of.
agent.track_vars = ['aNrmNow','cNrmNow', 'pLvlNow',
                    't_age', 'RiskyShareNow','mNrmNow']


# Run the simulations
agent.initializeSim()
agent.simulate()

raw_data = {'Age': agent.t_age_hist.flatten()+time_params['Age_born'] - 1,
            'pIncome': agent.pLvlNow_hist.flatten(),
            'rShare': agent.RiskyShareNow_hist.flatten(),
            'nrmM': agent.mNrmNow_hist.flatten(),
            'nrmC': agent.cNrmNow_hist.flatten()}

Data = pd.DataFrame(raw_data)
Data['Cons'] = Data.nrmC * Data.pIncome
Data['M'] = Data.nrmM * Data.pIncome

# Find the mean of each variable at every age
AgeMeans = Data.groupby(['Age']).mean().reset_index()

# %% Wealth income and consumption

plt.figure()
plt.plot(AgeMeans.Age, AgeMeans.pIncome,
         label = 'Income')
plt.plot(AgeMeans.Age, AgeMeans.M,
         label = 'Market resources')
plt.plot(AgeMeans.Age, AgeMeans.Cons,
         label = 'Consumption')
plt.legend()
plt.xlabel('Age')
plt.title('Variable Means Conditional on Survival')
plt.grid()

# Save figure
figname = 'YMC_Means'
plt.savefig(os.path.join(FigPath, figname + '.png'))
plt.savefig(os.path.join(FigPath, figname + '.jpg'))
plt.savefig(os.path.join(FigPath, figname + '.pdf'))
plt.savefig(os.path.join(FigPath, figname + '.svg'))

plt.ioff()
plt.draw()
plt.pause(1)

# %% Risky Share

# Find age percentiles
AgePC5 = Data.groupby(['Age']).quantile(0.05).reset_index()
AgePC95 = Data.groupby(['Age']).quantile(0.95).reset_index()

plt.figure()
plt.ylim([0, 1.1])
plt.plot(AgeMeans.Age, AgeMeans.rShare, label = 'Mean')
plt.plot(AgePC5.Age, AgePC5.rShare, '--k')
plt.plot(AgePC95.Age, AgePC95.rShare, '--k', label = 'Perc. 5 and 95')
plt.legend()

plt.xlabel('Age')
plt.ylabel('Risky Share')
plt.title('Risky Portfolio Share Mean Conditional on Survival')
plt.grid()

# Save figure
figname = 'RShare_Means'
plt.savefig(os.path.join(FigPath, figname + '.png'))
plt.savefig(os.path.join(FigPath, figname + '.jpg'))
plt.savefig(os.path.join(FigPath, figname + '.pdf'))
plt.savefig(os.path.join(FigPath, figname + '.svg'))

plt.ioff()
plt.draw()
plt.pause(1)
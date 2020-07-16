# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:10:36 2019

@author: mateo
"""

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
                    't_age', 'ShareNow','mNrmNow']


# Run the simulations
agent.initializeSim()
agent.simulate()

raw_data = {'Age': agent.history['t_age'].flatten()+time_params['Age_born'] - 1,
            'pIncome': agent.history['pLvlNow'].flatten(),
            'rShare': agent.history['ShareNow'].flatten(),
            'nrmM': agent.history['mNrmNow'].flatten(),
            'nrmC': agent.history['cNrmNow'].flatten()}

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

# plot till death - 1  
age_1 = time_params['Age_death'] - time_params['Age_born']

plt.figure()
plt.ylim([0, 1.1])
plt.plot(AgeMeans.Age[:age_1], AgeMeans.rShare[:age_1], label = 'Mean')
plt.plot(AgePC5.Age[:age_1], AgePC5.rShare[:age_1], '--r', label='Perc. 5')
plt.plot(AgePC95.Age[:age_1], AgePC95.rShare[:age_1], '--g', label = 'Perc. 95')
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


# %% Risky Share with 100-age rule

# Find age percentiles
AgePC5 = Data.groupby(['Age']).quantile(0.05).reset_index()
AgePC95 = Data.groupby(['Age']).quantile(0.95).reset_index()

plt.figure()
plt.ylim([0, 1.1])
plt.plot(AgeMeans.Age[:age_1], AgeMeans.rShare[:age_1], label = 'Mean')
plt.plot(AgePC5.Age[:age_1], AgePC5.rShare[:age_1], '--r', label='Perc. 5')
plt.plot(AgePC95.Age[:age_1], AgePC95.rShare[:age_1], '--g', label = 'Perc. 95')
# 100 age rule
x = range(time_params['Age_born'], time_params['Age_death'])
y = range(100 - time_params['Age_death'] + 1, 100 - time_params['Age_born'] + 1)[::-1]
y = np.array(y)/100
plt.plot(x, y, '--', color='orange', label = '100-age rule')
plt.legend()

plt.xlabel('Age')
plt.ylabel('Risky Share')
plt.title('Risky Portfolio Share Mean Conditional on Survival')
plt.grid()

# Save figure
figname = 'RShare_Means_100_age'
plt.savefig(os.path.join(FigPath, figname + '.png'))
plt.savefig(os.path.join(FigPath, figname + '.jpg'))
plt.savefig(os.path.join(FigPath, figname + '.pdf'))
plt.savefig(os.path.join(FigPath, figname + '.svg'))

plt.ioff()
plt.draw()
plt.pause(1)


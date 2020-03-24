# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:56:06 2019

@author: Mateo
"""

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import matplotlib.pyplot as plt
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
from Calibration.params import dict_portfolio, time_params, norm_factor, age_plot_params

agent = cpm.PortfolioConsumerType(**dict_portfolio)
agent.solve()

# %%
# Plot portfolio rule
eevalgrid = np.linspace(0,300,100)

# In the last period of life you consume everything
# so portfolio choice is irrelevant

# Ages
ages = age_plot_params
age_born = time_params['Age_born']
plt.figure()
for a in ages:
    plt.plot(eevalgrid,
             agent.solution[a-age_born].ShareFuncAdj(eevalgrid/norm_factor[a-age_born]),
             label = 'Age = %i' %(a))
plt.xlabel('Wealth')
plt.ylabel('Risky portfolio share')
plt.legend()
plt.grid()

# Save figure
figname = 'RShare_Pol'
plt.savefig(os.path.join(FigPath, figname + '.png'))
plt.savefig(os.path.join(FigPath, figname + '.jpg'))
plt.savefig(os.path.join(FigPath, figname + '.pdf'))
plt.savefig(os.path.join(FigPath, figname + '.svg'))

plt.ioff()
plt.draw()
plt.pause(1)

# Plot consumption function
plt.figure()
ages = age_plot_params
for a in ages:
    plt.plot(eevalgrid,
             agent.solution[a-age_born].cFuncAdj(eevalgrid/norm_factor[a-age_born])*norm_factor[a-age_born],
             label = 'Age = %i' %(a))
plt.xlabel('Wealth')
plt.ylabel('Consumption')
plt.legend()
plt.grid()

# Save figure
figname = 'Cons_Pol'
plt.savefig(os.path.join(FigPath, figname + '.png'))
plt.savefig(os.path.join(FigPath, figname + '.jpg'))
plt.savefig(os.path.join(FigPath, figname + '.pdf'))
plt.savefig(os.path.join(FigPath, figname + '.svg'))

plt.ioff()
plt.draw()
plt.pause(1)
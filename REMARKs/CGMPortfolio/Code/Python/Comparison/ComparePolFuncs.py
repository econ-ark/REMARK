# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:43:31 2019

@author: Mateo
"""

import numpy as np

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm

# Plotting tools
import matplotlib.pyplot as plt
import seaborn

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

# %% import Calibration
sys.path.append(my_file_path)
from Calibration.params import dict_portfolio, norm_factor

# %% Setup

# Path to fortran output
pathFort = os.path.join(my_file_path,"../Fortran/")

# Asset grid
npoints = 401
agrid = np.linspace(4,npoints+3,npoints)

# number of years
nyears = dict_portfolio['T_cycle']

# Initialize consumption, value, and share matrices
# (rows = age, cols = assets)
cons  = np.zeros((nyears, npoints))
val   = np.zeros((nyears, npoints))
share = np.zeros((nyears, npoints))

# %% Read and split policy functions
for year in range(nyears):
    
    y = year + 1
    if y < 10:
        ystring = '0' + str(y)
    else:
        ystring = str(y)
        
    rawdata = np.loadtxt(pathFort + 'year' + ystring + '.txt')
    
    share[year,:] = rawdata[range(npoints)]
    cons[year,:]  = rawdata[range(npoints,2*npoints)]
    val[year,:]   = rawdata[range(2*npoints,3*npoints)]
    
# %% Compute HARK's policy functions and store them in the same format
agent = cpm.PortfolioConsumerType(**dict_portfolio)
agent.solve()

# CGM's fortran code does not output the policy functions for the final period.
# thus len(agent.solve) = nyears + 1

# Initialize HARK's counterparts to the policy function matrices
h_cons  = np.zeros((nyears, npoints))
h_share = np.zeros((nyears, npoints))

# Fill with HARK's interpolated policy function at the required points
for year in range(nyears):
    
    h_cons[year,:]  = agent.solution[year].cFuncAdj(agrid/norm_factor[year])*norm_factor[year]
    h_share[year,:] = agent.solution[year].ShareFuncAdj(agrid/norm_factor[year])

# %% Compare the results
cons_error   = h_cons - cons
share_error = h_share - share

## Heatmaps

# Consumption

# Find max consumption (for the color map)
cmax = max(np.max(h_cons),np.max(cons))
f, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=True)
seaborn.despine(left=True)

seaborn.heatmap(h_cons, ax = axes[0], vmin = 0, vmax = cmax)
axes[0].set_title('HARK')
axes[0].set_xlabel('Assets', labelpad = 10)
axes[0].set_ylabel('Age')

seaborn.heatmap(cons, ax = axes[1], vmin = 0, vmax = cmax)
axes[1].set_title('CGM')
axes[1].set_xlabel('Assets', labelpad = 10)
axes[1].set_ylabel('Age')

seaborn.heatmap(cons_error, ax = axes[2], center = 0)
axes[2].set_title('HARK - CGM')
axes[2].set_xlabel('Assets', labelpad = 10)
axes[2].set_ylabel('Age')

f.suptitle('$C(\cdot)$')

f.tight_layout(rect=[0, 0.027, 1, 0.975])
f.subplots_adjust(top=0.85)

# Save figure
figname = 'Cons_Pol_Compare'
plt.savefig(os.path.join(FigPath, figname + '.png'))
plt.savefig(os.path.join(FigPath, figname + '.jpg'))
plt.savefig(os.path.join(FigPath, figname + '.pdf'))
plt.savefig(os.path.join(FigPath, figname + '.svg'))

plt.ioff()
plt.draw()
plt.pause(1)

# Risky share
f, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=True)
seaborn.despine(left=True)

seaborn.heatmap(h_share, ax = axes[0], vmin = 0, vmax = 1)
axes[0].set_title('HARK')
axes[0].set_xlabel('Assets', labelpad = 10)
axes[0].set_ylabel('Age')

seaborn.heatmap(share, ax = axes[1], vmin = 0, vmax = 1)
axes[1].set_title('CGM')
axes[1].set_xlabel('Assets', labelpad = 10)
axes[1].set_ylabel('Age')

seaborn.heatmap(share_error, ax = axes[2], center = 0)
axes[2].set_title('HARK - CGM')
axes[2].set_xlabel('Assets', labelpad = 10)
axes[2].set_ylabel('Age')

f.suptitle('$S(\cdot)$')

f.tight_layout(rect=[0, 0.027, 1, 0.975])
f.subplots_adjust(top=0.85)

# Save figure
figname = 'RShare_Pol_Compare'
plt.savefig(os.path.join(FigPath, figname + '.png'))
plt.savefig(os.path.join(FigPath, figname + '.jpg'))
plt.savefig(os.path.join(FigPath, figname + '.pdf'))
plt.savefig(os.path.join(FigPath, figname + '.svg'))

plt.ioff()
plt.draw()
plt.pause(1)
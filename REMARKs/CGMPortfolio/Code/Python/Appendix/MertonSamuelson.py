# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:31:45 2019

@author: Matt
"""

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import matplotlib.pyplot as plt
import numpy as np

from copy import copy
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

# %% Setup

# Import parameters from external file
sys.path.append(my_file_path) 
# Loading the parameters from the ../Code/Calibration/params.py script
from Calibration.params import dict_portfolio, time_params, norm_factor, age_plot_params

# A function to compute the analytical Merton Samuelson limit of the risky
# share
def RiskyShareMertSamLogNormal(RiskPrem, CRRA, RiskyVar):
    return RiskPrem/(CRRA*RiskyVar)*np.ones_like(1)


# Create a grid of market resources for the plots
    
mMin = 0    # Minimum ratio of assets to income to plot
mMax = 1e4 # Maximum ratio of assets to income to plot
mPts = 1000 # Number of points to plot 

eevalgrid = np.linspace(0,mMax,mPts) # range of values of assets for the plot

# Number of points that will be used to approximate the risky distribution
risky_count_grid = [5,50]

# %% Calibration and solution

for rcount in risky_count_grid:
    
    # Create a new dictionary and replace the number of points that
    # approximate the risky return distribution
    
    # Create new dictionary
    merton_dict = copy(dict_portfolio)
    merton_dict['RiskyCount'] = rcount

    # Create and solve agent
    agent = cpm.PortfolioConsumerType(**merton_dict)
    agent.solve()

    # Compute the analytical Merton-Samuelson limiting portfolio share
    RiskyVar = agent.RiskyStd**2
    RiskPrem = agent.RiskyAvg - agent.Rfree
    MS_limit = RiskyShareMertSamLogNormal(RiskPrem, agent.CRRA, RiskyVar)
    
    # Now compute the limiting share numerically, using the approximated
    # distribution
    agent.updateShareLimit()
    NU_limit = agent.ShareLimit
    
    # Plot by ages
    ages = age_plot_params
    age_born = time_params['Age_born']
    plt.figure()
    for a in ages:
        plt.plot(eevalgrid,
                 agent.solution[a-age_born].ShareFuncAdj(eevalgrid/norm_factor[a-age_born]),
                 label = 'Age = %i' %(a))
        
    plt.axhline(MS_limit, c='k', ls='--', label = 'M&S Limit')
    plt.axhline(NU_limit, c='k', ls='-.', label = 'Numer. Limit')

    plt.ylim(0,1.05)
    plt.xlim(eevalgrid[0],eevalgrid[-1])
    plt.legend()
    plt.title('Risky Portfolio Share by Age\n Risky distribution with {points} equiprobable points'.format(points = rcount))
    plt.xlabel('Wealth (m)')

    # Save figure
    figname = 'Merton_Samuelson_Limit_{points}'.format(points = rcount)
    plt.savefig(os.path.join(FigPath, figname + '.png'))
    plt.savefig(os.path.join(FigPath, figname + '.jpg'))
    plt.savefig(os.path.join(FigPath, figname + '.pdf'))
    plt.savefig(os.path.join(FigPath, figname + '.svg'))

    plt.ioff()
    plt.draw()
    plt.pause(1)

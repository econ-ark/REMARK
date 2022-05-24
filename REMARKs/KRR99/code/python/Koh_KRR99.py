#!/usr/bin/env python
# coding: utf-8

# Import IndShockConsumerType
import HARK
import sys
from HARK.ConsumptionSaving.ConsAggShockModel import (AggShockConsumerType, CobbDouglasEconomy, init_agg_shocks,
    init_cobb_douglas,
    solveConsAggShock,
    AggregateSavingRule
)
from ConsAggShockModel_tax import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
from HARK.utilities import plot_funcs, make_figs
import statsmodels.api as sm
from time import process_time
def mystr(number):
    return "{:.4f}".format(number)




# **Krusell and Rios-Rull's parameters for income and wealth distribution:**
# 
# Sorted by Wealth
# 
# <p>Group 1 (49%): Household wealth = 0.30, Earnings = 0.57<br>
# Group 2 (2%): Household wealth = 1, Earnings = 1<br>
# Group 3 (49%): Household wealth = 4.78, Earnings = 1.91</p>
# 
# <p>Sorted by Earnings<br>
# Group 1 (49%): Household wealth = 0.55, Earnings = 0.24<br>
# Group 2 (2%): Household wealth = 1, Earnings = 1<br>
# Group 3 (49%): Household wealth = 2.93, Earnings = 2.94</p>
# 
# *Greater inequality in household wealth than in household earnings*

# **Parameters in KRR99**
# <p>beta = 0.96 (Annual discount rate)<br>
# alpha = 0.429 (Utility weight of consumption)<br>
# sigma = 4 (Risk aversion parameter)<br>
# delta = 0.05 (Depreciation rate)<br>
# theta = 0.36 (Share of capital income)<br>
# Wealth-output ratio = 3.3<br>
# r = 0.06 (Pre-tax)<br>
# C/Y = 0.638 (Consumption-output ratio)<br>
# g/Y = 0.199 (Government spending-output ratio)<br>
# N = 0.34 (Labor hours)</p>



# Define a dictionary with calibrated parameters
AgentParameters = {
    "CRRA": 4.00,                    # Coefficient of relative risk aversion
    "DiscFac": 0.96,             # Default intertemporal discount factor; dummy value, will be overwritten
    "Rfree": 1.06, # Survival probability,
    "PermShkCount" : 1,                    # Number of points in discrete approximation to permanent income shocks - no shocks of this kind!
    "TranShkCount" : 3,                    # Number of points in discrete approximation to transitory income shocks - no shocks of this kind!
    "PermShkStd" : [0.0],                   # Standard deviation of log permanent income shocks - no shocks of this kind!
    "TranShkStd" : [0.5],                   # Standard deviation of log transitory income shocks - no shocks of this kind!
    "UnempPrb" : 0.0,                      # Probability of unemployment while working - no shocks of this kind!
    "UnempPrbRet" : 0.00,                  # Probability of "unemployment" while retired - no shocks of this kind!
    "IncUnemp" : 0.0,                      # Unemployment benefits replacement rate
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "PermGroFac" : [1.0],                  # Permanent income growth factor

    "CubicBool":False,
    "vFuncBool":True,
    "aXtraMin":0.00001,  # Minimum end-of-period assets in grid
    "aXtraMax":40,  # Maximum end-of-period assets in grid
    "aXtraCount":32,  # Number of points in assets grid
    "aXtraExtra":[None],
    "aXtraNestFac":3,  # Number of times to 'exponentially nest' when constructing assets grid
    "LivPrb":[1.0],  # Survival probability
    # "LivPrb":[1.0 - 1.0/160.0],  # Survival probability

    "cycles":0,
    "T_cycle":1,
    'T_sim':2,  # Number of periods to simulate (idiosyncratic shocks model, perpetual youth)
    'T_age': 100,
    'IndL': 1.0,  # Labor supply per individual (constant)
    'aNrmInitMean':np.log(0.00001),
    'aNrmInitStd':0.0,
    'pLvlInitMean':0.0,
    'pLvlInitStd':0.0,
    'AgentCount':100,
    'MgridBase': np.array([0.1,0.3,0.6,
                           0.8,0.9,0.98,
                           1.0,1.02,1.1,
                           1.2,1.6,2.0,
                           3.0]),          # Grid of capital-to-labor-ratios (factors)
    'PermGroFacAgg': 1.0,

    # Variables necessary for AggShockConsumerType_tax model
    'tax_rate':0.00,

    # Parameters describing the income process
    # New Parameters that we need now
    'PermShkAggStd' : [0.0],           # Standard deviation of log aggregate permanent shocks by state. No continous shocks in a state.
    'TranShkAggStd' : [0.0],           # Standard deviation of log aggregate transitory shocks by state. No continuous shocks in a state.
}


# Code source: https://github.com/econ-ark/HARK/blob/master/examples/ConsumptionSaving/example_ConsAggShockModel.ipynb
# See also, on HARK.core.Market: https://hark.readthedocs.io/en/latest/reference/tools/core.html#HARK.core.Market
# For details on Krusell-Smith model in HARK: https://github.com/econ-ark/KrusellSmith/blob/master/Code/Python/KrusellSmith.ipynb
# See also: https://github.com/econ-ark/HARK/blob/master/examples/HowWeSolveIndShockConsumerType/HowWeSolveIndShockConsumerType.ipynb

# Define aggregate shock consumer (simplest consumer type possible that has both labor and assets/capital)
AggShockAgent_tax = AggShockConsumerType_tax(**AgentParameters)


# Create Cobb-Douglas economy with AggShockExample consumers
# EconomyExample = CobbDouglasEconomy(agents=[AggShockAgent_tax], **EconomyDictionary)
# Somehow the above method of applying parameters like the above line doesn't seem to work

EconomyExample = CobbDouglasEconomy_tax(agents=[AggShockAgent_tax], PermShkAggCount = 1, TranShkAggCount = 1,
                                    PermShkAggStd = 0.0, TranShkAggStd = 0.0, DeprFac = 0.05, PermGroFacAgg = 1.0,
                                    AggregateL = 1.0, CapShare = 0.36, CRRA = 4.0, tolerance = 0.01, tax_rate = 0.00)


# Have the consumers inherit relevant objects from the economy
AggShockAgent_tax.get_economy_data(EconomyExample)

# Simulate a history of aggregate shocks
EconomyExample.make_AggShkHist()

# Solve for aggregate shock consumer model
AggShockAgent_tax.solve()
AggShockAgent_tax.track_vars = ['pLvl','TranShk']

# Solve the "macroeconomic" model by searching for a "fixed point dynamic rule"
print("Now solving for the equilibrium of a Cobb-Douglas economy.  This might take a few minutes...")
EconomyExample.solve()

AggShockAgent_tax.unpack('cFunc')

print("Consumption function at each aggregate market resources-to-labor ratio gridpoint:")
m_grid = np.linspace(0, 5, 200)
for M in AggShockAgent_tax.Mgrid.tolist():
    mMin = AggShockAgent_tax.solution[0].mNrmMin(M)
    c_at_this_M = AggShockAgent_tax.cFunc[0](m_grid + mMin, M * np.ones_like(m_grid))
    plt.plot(m_grid + mMin, c_at_this_M)
plt.show()

AggShockAgent_tax.unpack('vFunc')

print("Value function at each aggregate market resources-to-labor ratio gridpoint:")
m_grid = np.linspace(0, 5, 200)
for M in AggShockAgent_tax.Mgrid.tolist():
    mMin = AggShockAgent_tax.solution[0].mNrmMin(M)+0.5
    v_at_this_M = AggShockAgent_tax.vFunc[0](m_grid + mMin, M * np.ones_like(m_grid))
    plt.plot(m_grid + mMin, v_at_this_M)
plt.show()

# Normalized market resources of each agent
sim_market_resources = AggShockAgent_tax.state_now['mNrm']

# Normalized assets of each agent
sim_wealth = AggShockAgent_tax.state_now['aNrm']

# Summary Statistics

# Lump-sum transfers are calculated through AggShockConsumerType_tax.calc_transfers() method:
print("The lump-sum transfer in terms of permanent income is: " + str(AggShockAgent_tax.calc_transfers()))

print("The mean of individual market resources is " + str(sim_market_resources.mean()) + "; the standard deviation is "
      + str(sim_market_resources.std()) + "; the median is " + str(np.median(sim_market_resources)) + ".")
print("The mean of individual wealth is " + str(sim_wealth.mean()) + "; the standard deviation is "
      + str(sim_wealth.std()) + "; the median is " + str(np.median(sim_wealth)) + ".")

print("The median level of market resources is: " + str(np.median(AggShockAgent_tax.state_now['mNrm'])))

# Lorenz Curve of Wealth Distribution

from HARK.datasets import load_SCF_wealth_weights
from HARK.utilities import get_lorenz_shares, get_percentiles

SCF_wealth, SCF_weights = load_SCF_wealth_weights()

pctiles = np.linspace(0.001,0.999,200)

SCF_Lorenz_points = get_lorenz_shares(SCF_wealth,weights=SCF_weights,percentiles=pctiles)
sim_Lorenz_points = get_lorenz_shares(sim_wealth,percentiles=pctiles)
plt.plot(pctiles,pctiles,'-r')
plt.plot(pctiles,SCF_Lorenz_points,'--k')
plt.plot(pctiles,sim_Lorenz_points,'-b')
plt.xlabel('Percentile of net worth')
plt.ylabel('Cumulative share of wealth')
plt.show(block=False)

rates = 20
tax_rates = np.linspace(0.00, 0.95, num=rates)

v_at_p90_wealth = [] # Vector for value function of 90th percentile wealth agent at each level of flat income tax rate
v_at_p80_wealth = [] # Vector for value function of 80th percentile wealth agent at each level of flat income tax rate
v_at_p70_wealth = [] # Vector for value function of 70th percentile wealth agent at each level of flat income tax rate
v_at_p60_wealth = [] # Vector for value function of 60th percentile wealth agent at each level of flat income tax rate
v_at_median_wealth = [] # Vector for value function of median wealth agent at each level of flat income tax rate
v_at_p40_wealth = [] # Vector for value function of 40th percentile wealth agent at each level of flat income tax rate
v_at_p30_wealth = [] # Vector for value function of 30th percentile wealth agent at each level of flat income tax rate
v_at_p20_wealth = [] # Vector for value function of 20th percentile wealth agent at each level of flat income tax rate
v_at_p10_wealth = [] # Vector for value function of 10th percentile wealth agent at each level of flat income tax rate

for tau in tax_rates:

    AggShockAgent_tax_tau = deepcopy(AggShockAgent_tax)
    AggShockAgent_tax_tau.tax_rate = tau
    EconomyExample_tau = deepcopy(EconomyExample)
    EconomyExample_tau.tax_rate = tau
    AggShockAgent_tax_tau.get_economy_data(EconomyExample_tau)
    EconomyExample_tau.make_AggShkHist()
    AggShockAgent_tax_tau.solve()
    AggShockAgent_tax_tau.initialize_sim()
    AggShockAgent_tax_tau.simulate()
    AggShockAgent_tax_tau.track_vars = ['aNrm','pLvl','mNrm','TranShk']
    EconomyExample_tau.solve()
    AggShockAgent_tax_tau.unpack('vFunc')

    sim_market_resources_tau = AggShockAgent_tax_tau.state_now['mNrm']
    sim_wealth_tau = AggShockAgent_tax_tau.state_now['aNrm']
    
    print("The flat income tax rate is: " + mystr(AggShockAgent_tax_tau.tax_rate))

    print("The lump-sum transfer in terms of permanent income is: " + mystr(AggShockAgent_tax_tau.calc_transfers()))

    print("The mean of individual wealth is " + mystr(sim_wealth_tau.mean()) + "; the standard deviation is "
          + str(sim_wealth_tau.std()) + "; the median is " + mystr(np.median(sim_wealth_tau)) + ".")
    print("The mean of individual market resources is " + mystr(sim_market_resources_tau.mean()) + "; the standard deviation is "
          + str(sim_market_resources_tau.std()) + "; the median is " + mystr(np.median(sim_market_resources_tau)) + ".")
    print("The 90th percentile of individual wealth is " + mystr(np.percentile(sim_wealth_tau,90)) + ".")
    print("The 80th percentile of individual wealth is " + mystr(np.percentile(sim_wealth_tau,80)) + ".")
    print("The 70th percentile of individual wealth is " + mystr(np.percentile(sim_wealth_tau,70)) + ".")
    print("The 60th percentile of individual wealth is " + mystr(np.percentile(sim_wealth_tau,60)) + ".")
    print("The median of individual wealth is " + mystr(np.median(sim_wealth_tau)) + ".")
    print("The 40th percentile of individual wealth is " + mystr(np.percentile(sim_wealth_tau,40)) + ".")
    print("The 30th percentile of individual wealth is " + mystr(np.percentile(sim_wealth_tau,30)) + ".")
    print("The 20th percentile of individual wealth is " + mystr(np.percentile(sim_wealth_tau,20)) + ".")
    print("The 10th percentile of individual wealth is " + mystr(np.percentile(sim_wealth_tau,10)) + ".\n")
    
    # Tax rate as determined by agent (pre-tax)'s wealth and income
    sim_p90_wealth_tau = np.percentile(sim_wealth_tau,90)
    sim_p90_market_resources_tau = np.percentile(sim_market_resources_tau,90)
    sim_p80_wealth_tau = np.percentile(sim_wealth_tau,80)
    sim_p80_market_resources_tau = np.percentile(sim_market_resources_tau,80)
    sim_p70_wealth_tau = np.percentile(sim_wealth_tau,70)
    sim_p70_market_resources_tau = np.percentile(sim_market_resources_tau,70)
    sim_p60_wealth_tau = np.percentile(sim_wealth_tau,60)
    sim_p60_market_resources_tau = np.percentile(sim_market_resources_tau,60)
    sim_median_wealth_tau = np.median(sim_wealth_tau)
    sim_median_market_resources_tau = np.median(sim_market_resources_tau)
    sim_p40_wealth_tau = np.percentile(sim_wealth_tau,40)
    sim_p40_market_resources_tau = np.percentile(sim_market_resources_tau,40)
    sim_p30_wealth_tau = np.percentile(sim_wealth_tau,30)
    sim_p30_market_resources_tau = np.percentile(sim_market_resources_tau,30)
    sim_p20_wealth_tau = np.percentile(sim_wealth_tau,20)
    sim_p20_market_resources_tau = np.percentile(sim_market_resources_tau,20)
    sim_p10_wealth_tau = np.percentile(sim_wealth_tau,10)
    sim_p10_market_resources_tau = np.percentile(sim_market_resources_tau,10)
    # Find value function of post-tax Xth-percentile wealth agent/voter, with X taking values from 10 to 90
    # vFunc arguments: Each agent's level of market resources
    # and median agent's capital-labor ratio (assumed as 1.0 for now)
    v_at_p90_wealth_tau = AggShockAgent_tax_tau.vFunc[0](sim_p90_market_resources_tau, 1.0)
    v_at_p80_wealth_tau = AggShockAgent_tax_tau.vFunc[0](sim_p80_market_resources_tau, 1.0)
    v_at_p70_wealth_tau = AggShockAgent_tax_tau.vFunc[0](sim_p70_market_resources_tau, 1.0)
    v_at_p60_wealth_tau = AggShockAgent_tax_tau.vFunc[0](sim_p60_market_resources_tau, 1.0)
    v_at_median_wealth_tau = AggShockAgent_tax_tau.vFunc[0](sim_median_market_resources_tau, 1.0)
    v_at_p40_wealth_tau = AggShockAgent_tax_tau.vFunc[0](sim_p40_market_resources_tau, 1.0)
    v_at_p30_wealth_tau = AggShockAgent_tax_tau.vFunc[0](sim_p30_market_resources_tau, 1.0)
    v_at_p20_wealth_tau = AggShockAgent_tax_tau.vFunc[0](sim_p20_market_resources_tau, 1.0)
    v_at_p10_wealth_tau = AggShockAgent_tax_tau.vFunc[0](sim_p10_market_resources_tau, 1.0)

    v_at_p90_wealth.append(v_at_p90_wealth_tau)
    v_at_p80_wealth.append(v_at_p80_wealth_tau)
    v_at_p70_wealth.append(v_at_p70_wealth_tau)
    v_at_p60_wealth.append(v_at_p60_wealth_tau)
    v_at_median_wealth.append(v_at_median_wealth_tau)
    v_at_p40_wealth.append(v_at_p40_wealth_tau)
    v_at_p30_wealth.append(v_at_p30_wealth_tau)
    v_at_p20_wealth.append(v_at_p20_wealth_tau)
    v_at_p10_wealth.append(v_at_p10_wealth_tau)

# Create graph of value function of agent with median level of wealth for each tax rate from 0.00 to 0.95 (in increments of 0.05)
print(v_at_median_wealth)
print(mystr(np.max(v_at_median_wealth)))
optimal_tax_rate = tax_rates[v_at_median_wealth.index(np.max(v_at_median_wealth))]
print("The optimal tax rate for the median voter is " + str(mystr(optimal_tax_rate)) + ".")


plt.figure(figsize=(7,4))
plt.plot(tax_rates, v_at_median_wealth, 'b-', label = 'Value function of median wealth agent')
plt.xlabel('Flat income tax rate')
plt.ylabel('Value function of median wealth agent')
plt.show()




plt.figure(figsize=(13,6))
plt.plot(tax_rates, v_at_p90_wealth, 'r--', label = 'Value function of p90 wealth agent')
plt.plot(tax_rates, v_at_p80_wealth, 'g--', label = 'Value function of p80 wealth agent')
plt.plot(tax_rates, v_at_p70_wealth, 'b--', label = 'Value function of p70 wealth agent')
plt.plot(tax_rates, v_at_p60_wealth, 'y--', label = 'Value function of p60 wealth agent')
plt.plot(tax_rates, v_at_median_wealth, 'k-', label = 'Value function of median wealth agent')
plt.plot(tax_rates, v_at_p40_wealth, 'r-', label = 'Value function of p40 wealth agent')
plt.plot(tax_rates, v_at_p30_wealth, 'g-', label = 'Value function of p30 wealth agent')
plt.plot(tax_rates, v_at_p20_wealth, 'b-', label = 'Value function of p20 wealth agent')
plt.plot(tax_rates, v_at_p10_wealth, 'y-', label = 'Value function of p10 wealth agent')

plt.xlabel('Flat income tax rate')
plt.ylabel('Value function of agents')
plt.legend()
plt.show()




optimal_tax_rate_p90 = tax_rates[v_at_p90_wealth.index(np.max(v_at_p90_wealth))]
optimal_tax_rate_p80 = tax_rates[v_at_p80_wealth.index(np.max(v_at_p80_wealth))]
optimal_tax_rate_p70 = tax_rates[v_at_p70_wealth.index(np.max(v_at_p70_wealth))]
optimal_tax_rate_p60 = tax_rates[v_at_p60_wealth.index(np.max(v_at_p60_wealth))]
optimal_tax_rate_median = tax_rates[v_at_median_wealth.index(np.max(v_at_median_wealth))]
optimal_tax_rate_p40 = tax_rates[v_at_p40_wealth.index(np.max(v_at_p40_wealth))]
optimal_tax_rate_p30 = tax_rates[v_at_p30_wealth.index(np.max(v_at_p30_wealth))]
optimal_tax_rate_p20 = tax_rates[v_at_p20_wealth.index(np.max(v_at_p20_wealth))]
optimal_tax_rate_p10 = tax_rates[v_at_p10_wealth.index(np.max(v_at_p10_wealth))]

print("The optimal tax rate for the 90th percentile voter is " + str(mystr(optimal_tax_rate_p90)) + ".")
print("The optimal tax rate for the 80th percentile voter is " + str(mystr(optimal_tax_rate_p80)) + ".")
print("The optimal tax rate for the 70th percentile voter is " + str(mystr(optimal_tax_rate_p70)) + ".")
print("The optimal tax rate for the 60th percentile voter is " + str(mystr(optimal_tax_rate_p60)) + ".")
print("The optimal tax rate for the median voter is " + str(mystr(optimal_tax_rate_median)) + ".")
print("The optimal tax rate for the 40th percentile voter is " + str(mystr(optimal_tax_rate_p40)) + ".")
print("The optimal tax rate for the 30th percentile voter is " + str(mystr(optimal_tax_rate_p30)) + ".")
print("The optimal tax rate for the 20th percentile voter is " + str(mystr(optimal_tax_rate_p20)) + ".")
print("The optimal tax rate for the 10th percentile voter is " + str(mystr(optimal_tax_rate_p10)) + ".")


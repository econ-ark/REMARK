#!/usr/bin/env python
# coding: utf-8

# Import IndShockConsumerType
import HARK
import sys
from HARK.ConsumptionSaving.ConsAggShockModel import (AggShockConsumerType,
                                                      CobbDouglasEconomy)
from ConsAggShockModel_tax import (AggShockConsumerType_tax, CobbDouglasEconomy_tax)
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
AggShockAgent_tax.track_vars = ['aNrm','pLvl','mNrm','cNrm','TranShk']


# Solve the "macroeconomic" model by searching for a "fixed point dynamic rule"
print("Now solving for the equilibrium of a Cobb-Douglas economy.  This might take a few minutes...")
EconomyExample.solve()


# AggShockAgent_tax.unpack('cFunc')
#
# print("Consumption function at each aggregate market resources-to-labor ratio gridpoint:")
# m_grid = np.linspace(0, 10, 200)
# for M in AggShockAgent_tax.Mgrid.tolist():
#     mMin = AggShockAgent_tax.solution[0].mNrmMin(M)
#     c_at_this_M = AggShockAgent_tax.cFunc[0](m_grid + mMin, M * np.ones_like(m_grid))
#     plt.plot(m_grid + mMin, c_at_this_M)
#     plt.ylim(0.0, 3.5)
# plt.show()
#
# AggShockAgent_tax.unpack('vFunc')
#
# print("Value function at each aggregate market resources-to-labor ratio gridpoint:")
# m_grid = np.linspace(0, 10, 200)
# for M in AggShockAgent_tax.Mgrid.tolist():
#     mMin = AggShockAgent_tax.solution[0].mNrmMin(M)+0.5
#     v_at_this_M = AggShockAgent_tax.vFunc[0](m_grid + mMin, M * np.ones_like(m_grid))
#     plt.plot(m_grid + mMin, v_at_this_M)
# plt.show()
#
#
# # Code for finding optimal tax rate given vFunc of median AggShockConsumerType agent in Cobb-Douglas Economy
#
# # Put this in loop in order to find optimal tax rate for median (wealth) voter

rates = 20
tax_rates = np.linspace(0.00, 0.95, num=rates)
v_at_median_wealth = []

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
    AggShockAgent_tax_tau.track_vars = ['aNrm','pLvl','mNrm','cNrm','TranShk']
    EconomyExample_tau.solve()
    AggShockAgent_tax_tau.unpack('vFunc')

    sim_market_resources_tau = AggShockAgent_tax_tau.state_now['mNrm']

    sim_wealth_tau = AggShockAgent_tax_tau.state_now['aNrm']

    print("The lump-sum transfer in terms of permanent income is: " + str(AggShockAgent_tax_tau.calc_transfers()))

    print("The mean of individual market resources is " + str(sim_market_resources_tau.mean()) + "; the standard deviation is "
          + str(sim_market_resources_tau.std()) + "; the median is " + str(np.median(sim_market_resources_tau)) + ".")
    print("The mean of individual wealth is " + str(sim_wealth_tau.mean()) + "; the standard deviation is "
          + str(sim_wealth_tau.std()) + "; the median is " + str(np.median(sim_wealth_tau)) + ".")

    print("The median level of market resources is: " + str(np.median(AggShockAgent_tax_tau.state_now['mNrm'])))
    print("And also with median wealth + income: " + str(np.median(sim_wealth_tau + sim_income_tau)))


    # print("The mean of post-tax individual income is " + str(sim_post_income_tau.mean()) + "; the standard deviation is "
    #       + str(sim_post_income_tau.std()) + "; the median is " + str(np.median(sim_post_income_tau)) + ".")
    # print("The aggregate pre-tax wealth-income ratio is " + str(sim_wealth_tau.mean() / sim_pre_income_tau.mean()) + ".")
    # print("The aggregate post-tax wealth-income ratio is " + str(sim_wealth_tau.mean() / sim_post_income_tau.mean()) + ".")

    # Tax rate as determined by median agent (pre-tax)'s wealth and income

    sim_median_wealth_tau = np.median(sim_wealth_tau)
    # sim_median_income_tau = np.median(sim_income_tau)
    sim_median_market_resources_tau = np.median(sim_market_resources_tau)

    # Find value function of median wealth agent/voter
    # vFunc arguments: Median agent's level of market resources
    # and median agent's capital-labor ratio (assumed as 1.0 for now)
    v_at_median_wealth_tau = AggShockAgent_tax_tau.vFunc[0](sim_median_market_resources_tau,
                                                            1.0)

    v_at_median_wealth.append(v_at_median_wealth_tau)

    print("The market resources of the median agent is "
          + str(mystr(sim_median_market_resources_tau))
          + ".\n The value function for the median-wealth voter at tax rate "
          + str(mystr(AggShockAgent_tax_tau.tax_rate))
          + " is " + str(mystr(v_at_median_wealth_tau)) + ".")


print(v_at_median_wealth)
print(mystr(np.max(v_at_median_wealth)))
optimal_tax_rate = tax_rates[v_at_median_wealth.index(np.max(v_at_median_wealth))]
print("The optimal tax rate for the median voter is " + str(mystr(optimal_tax_rate)) + ".")

plt.figure(figsize=(7,4))
plt.plot(tax_rates, v_at_median_wealth, 'b-', label = 'Value function of median wealth agent')
plt.xlabel('Flat income tax rate')
plt.ylabel('Value function of median wealth agent')
plt.show()

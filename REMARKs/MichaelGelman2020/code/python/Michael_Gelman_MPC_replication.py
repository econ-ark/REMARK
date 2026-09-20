# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # What drives heterogeneity in the MPC (circumstances view vs. characteristics view)
#
# Michael Gelman
#
# Presented by Jionglin (Andy) Zheng
#
# Python version: 3.8.5
#
# HARK version: 0.11.0

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Summary
#
# The paper discusses the relative importance of circumstances view and characteristics view in explaining the variance of the MPC
#   
# Main findings:  
# * the paper compares the average cash on hand distribution simulated in the model to the empirical distribution estimated from the data, and the model matches the empirical distribution fairly well
# * the paper finds that within- and across-individual differences in cash on hand play roughly equal roles in explaining MPC variance (characteristics variance share is about 0.45)

# %% [markdown] slideshow={"slide_type": "slide"}
# ![Figure15_comparison.png](attachment:Figure15_comparison.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## The Model
# ### Model description
# Optimization problem of individual $i$ solves the following utility maximization problem:
#
# \begin{eqnarray*}
# &\max_{\left\{C_{i j}\right\}_{j=t}^{\infty}}& \mathbb{E}_{t}\left[\sum_{j=t}^{\infty} \beta_{i}^{j-t} \frac{C_{i j}^{1-\theta}}{1-\theta}\right]\\
# \notag & \text{subject to}& \\
# &A_{i t+1}&=(1+r)\left(A_{i t}+Y_{i t}-C_{i t}\right) \\
# &A_{i t+1}& \geq b \\
# &Y_{i t}&=\bar{Y}_{i} \varepsilon_{i t} \\
# &\varepsilon_{i t}& \stackrel{i i d}{\sim} N\left(1, \sigma_{Y}^{2}\right)
# \end{eqnarray*}
#
# where $\beta_{i}, r, C_{it}, A_{it}$ and $Y_{it}$ represents the time discount factor, the interest rate, consumption, liquid assets, and income respectively.
#
#

# %% [markdown] slideshow={"slide_type": "slide"}
# The Bellman form of the value function for individuals is:
#
# \begin{eqnarray*}
# V\left(x_{i t}\right) &=& \max _{a_{i t+1}}\left\{u\left(c_{i t}\right)+\beta_{i} \mathbb{E}\left[V\left(x_{i t+1}\right)\right]\right\}
# \notag &\text{s.t.}&\\
# x_{i t+1} &=& (1+r)\left(x_{i t}-c_{i t}\right)+y_{i t+1},\\
# \end{eqnarray*}
# and previous constraints.
#
# Substituting in for $c_{i t}$ and $x_{i t+1}$ results in an equation in terms of $x_{i t}, a_{i t+1},$ and $y_{i t+1}$, where $x_{it}$ is cash-on-hand.
#
# \begin{align*}
# V\left(x_{i t}\right)=\max _{a_{i t+1}}\left\{u\left(x_{i t}-\frac{a_{i t+1}}{1+r}\right)+\beta_{i} \mathbb{E}\left[V\left(a_{i t+1}+y_{i t+1}\right)\right]\right\}
# \end{align*}

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Circumstances and characteristics view
# * temporary circumstances
#     - $\beta_i = \bar{\beta}$ 
# * persistent characteristics
#     - $\beta_i \sim \; U(\bar{\beta} - \Delta, \bar{\beta} + \Delta)$
# ![Figure01_comparison_two_views.png](attachment:Figure01_comparison_two_views.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### replication using HARK (figure 1)
# * calibration
# \begin{aligned}
# &\begin{array}{llll}
# \hline \text { Parameter } & \text { Value } & \text { Notes } & \text { Description } \\
# \hline u(x) & \frac{x^{1-\theta}}{1-\theta} & \text { CRRA utility } & \text { utility function } \\
# \theta & 1 & \text { standard } & \text { coefficient of relative risk aversion } \\
# \bar{\beta} & 0.9941 & & \text { average discount factor } \\
# \Delta & 0.0190 & 0 \text { for circumstance model } & \text { discount factor dispersion } \\
# \sigma_{Y} & 0.20 & \text { estimated from dataset } & \text { S.D. of temporary shocks } \\
# \text {refund}_{i t} & 0.60 & \text { estimated from dataset } & \text { average normalized refund } \\
# r & 0.01 / 12 & \text { monthly } r \text { on checking/saving } & \text { interest rate } \\
# b & 0 & \text { no borrowing condition } & \text { borrowing limit } \\
# \hline
# \end{array}\\
# &\text { Notes: The parameters correspond to a monthly frequency. }
# \end{aligned}

# %% [markdown]
# #### HARK result (figure 1)

# %% slideshow={"slide_type": "skip"}
import HARK
HARK.__version__

# This cell does some standard python setup

# Tools for navigating the filesystem
import sys
import os

import csv
import os.path

# Import related generic python packages
import numpy as np
import math
mystr = lambda number : "{:.4f}".format(number)
from copy import copy, deepcopy

# Plotting tools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show

import pandas as pd

# iPython gives us some interactive and graphical tools
from IPython import get_ipython # In case it was run from python instead of ipython

# The warnings package allows us to ignore some harmless but alarming warning messages
import warnings
warnings.filterwarnings("ignore")

from HARK.utilities import plot_funcs

# %% slideshow={"slide_type": "skip"}
# simulation for distribution of cash on hand
IdiosyncDict={
    # Parameters shared with the perfect foresight model
    "CRRA": 1.00000000000001,                           # Coefficient of relative risk aversion
    "Rfree": 1 + 0.01/12,                         # Interest factor on assets
    "DiscFac": 0.9941,                       # Intertemporal discount factor, monthly frequency
    "LivPrb" : 12*[1.0],                     # Survival probability, 0.99979
    "PermGroFac" : 12*[1.0],                  # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd" : 12*[0.0000000000000001],                  # Standard deviation of log permanent shocks to income
    "PermShkCount" : 0,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : 12*[0.2],                  # Standard deviation of log transitory shocks to income
    "TranShkCount" : 7,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.00,                     # Probability of unemployment while working
    "IncUnemp" : 0.3,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.000,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.0,                      # Flat income tax rate (legacy parameter, will be removed in future)

    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 20,                       # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 48,                     # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra" : [None],                 # Additional values to add to aXtraGrid

    # A few other paramaters
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool" : True,                    # Whether to calculate the value function during solution
    "CubicBool" : False,                   # Preference shocks currently only compatible with linear cFunc
    "T_cycle" : 12,                         # Number of periods in the cycle for this agent type
                                           # quarterly data

    # Parameters only used in simulation
    "AgentCount" : 1,                  # Number of agents of this type
    "T_sim" : 5000,                         # Number of periods to simulate
    "aNrmInitMean" : -6.0,                 # Mean of log initial assets
    "aNrmInitStd"  : 1.0,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
}

# %% slideshow={"slide_type": "slide"}
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType

IndShockSimExample = IndShockConsumerType(**IdiosyncDict)
IndShockSimExample.cycles = 0 # Make this type have an infinite horizon
IndShockSimExample.solve()

plot_funcs(IndShockSimExample.solution[0].cFunc,IndShockSimExample.solution[0].mNrmMin,5)

# simulation
IndShockSimExample.track_vars = ['aNrm','mNrm','cNrm','pLvl']
IndShockSimExample.initialize_sim()
IndShockSimExample.simulate()

# %% slideshow={"slide_type": "slide"}
# distribution of cash on hand
density = np.histogram(IndShockSimExample.history['mNrm'], density=True)#print(density)
n, bins, patches = plt.hist(IndShockSimExample.history['mNrm'], density = True)

# %% slideshow={"slide_type": "slide"}
df1 = pd.DataFrame(IndShockSimExample.history['mNrm'], columns = ['beta = 0.9941'] ) #Converting array to pandas DataFrame
df1.plot(kind = 'density', title = 'distribution of cash on hand')

# %% slideshow={"slide_type": "skip"}
# circumstances view
IndShockSimExample2 = deepcopy(IndShockSimExample)
IndShockSimExample2.DiscFac = 0.9979
IndShockSimExample2.update_income_process()
IndShockSimExample2.solve()

IndShockSimExample2.track_vars = ['aNrm','mNrm','cNrm','pLvl']
IndShockSimExample2.initialize_sim()
IndShockSimExample2.simulate()

# %% slideshow={"slide_type": "slide"}
density = np.histogram(IndShockSimExample2.history['mNrm'], density=True)
print(density)
n, bins, patches = plt.hist(IndShockSimExample2.history['mNrm'], density = True)

# %% slideshow={"slide_type": "slide"}
df2 = pd.DataFrame(IndShockSimExample2.history['mNrm'], columns = ['beta = 0.9979'] ) #Converting array to pandas DataFrame
df2.plot(kind = 'density', label = 'beta = 0.9941')

# %% slideshow={"slide_type": "slide"}
# plot two densities on a single figure & draw consumption funcs
ax = df1.plot(kind = 'density')
df2.plot(ax = ax, kind = 'density', title = 'distribution of cash on hand')

# Final consumption function c=m
m = np.linspace(0.1,1,100)
plot_funcs([IndShockSimExample.solution[0].cFunc, IndShockSimExample2.solution[0].cFunc],0.,5.)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### comparing HARK replication with Gelman figure 1
# ![Figure01_comparison_two_views.png](attachment:Figure01_comparison_two_views.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### HARK result (figure 2)

# %%
# write beta_list and target level wealth to file
IndShockSimExample3 = deepcopy(IndShockSimExample)
IndShockSimExample3.CRRA = 1.62
IndShockSimExample3.DiscFac = 0.99

'''
calculate target level of wealth to different betas

@param: 
    filename  : name of the file to store all the values; delimiter is set to be ' '
    startValue: x axis limit of beta; ending value is set to be 0.9989
'''
def generate_TBS_Lvl_to_beta(filename, startValue):
    f = open(filename, 'w')
    beta = startValue
    
    while (beta <= 0.9989):
        IndShockSimExample3.DiscFac = beta
        IndShockSimExample3.update_income_process()
        IndShockSimExample3.solve()

        targetLvlWealth = IndShockSimExample3.solution[0].mNrmTrg

        f.write(str(beta) + ' ')  # Write beta
        f.write(str(targetLvlWealth) + '\n')
        f.flush()
        
        # error message
        if(beta <= 0.998):
            beta += 0.00005
        elif((beta > 0.998) and (beta <=0.9988)):
            beta += 0.0015

    f.close()
    

'''
plot discount factor vs. target buffer stock, figure 2

@param: filename
'''
def plot_TBS_level_to_beta(filename):
    f = open(os.path.abspath(filename), 'r')
    my_reader = csv.reader(f, delimiter=' ')
    raw_data = list(my_reader)
    
    targetLvl = []
    beta_list = []
    
    for i in range(len(raw_data)):
        beta_list.append(float(raw_data[i][0]))
        targetLvl.append(float(raw_data[i][1]))
    beta_list = np.array(beta_list)
    targetLvl = np.array(targetLvl)
    f.close()

    plt.plot(beta_list, targetLvl, '-k', color='b', linewidth=1.5)
    plt.xlabel(r'discount factor $\beta$',fontsize=12)
    plt.ylabel('Target buffer stock',fontsize=12)
    plt.title('figure 2: target buffer stock and the discount factor',fontsize=12)
    plt.ylim(1.3, 2.0)
    #plt.savefig('./figure3_reproduction.pdf')
    plt.show()    


# %%
# plot discount factor vs. target buffer stock, figure 2
filename = 'TBS_Lvl_to_beta.txt'

try:
    if(os.path.isfile(filename)):
        plot_TBS_level_to_beta(filename)
    else:
        print('file does not exist. file will be generated to plot the graph.')
        generate_TBS_Lvl_to_beta(filename, 0.99)
        plot_TBS_level_to_beta(filename)
except OSError: 
    '''
    This exception is raised when a system function returns a system-related error, 
    including I/O failures such as “file not found” or “disk full” 
    (not for illegal argument types or other incidental errors).
    '''
    print('file error.')


# %% [markdown]
# ##### original figure in paper

# %% [markdown]
# ![Figure02_TBS_discount_factor.png](attachment:Figure02_TBS_discount_factor.png)

# %% slideshow={"slide_type": "slide"}
# figure 3
plt.plot(IndShockSimExample.history['mNrm'])
plt.xlabel('Time')
plt.ylabel('Individual cash on hand paths')
plt.xlim(150, 200)
plt.ylim(1.0, 1.9)
plt.axhline(y=IndShockSimExample.solution[0].mNrmTrg, color='b', linestyle='--') # target level of wealth
plt.title('figure 3: cash on hand time series',fontsize=12)
plt.show()

print(IndShockSimExample.solution[0].mNrmTrg) #value of (normalized) market resources m at which individual consumer expects m not to change

# %% [markdown]
# ##### original figure in paper

# %% [markdown]
# ![Figure03_cash_on_hand_time_series.png](attachment:Figure03_cash_on_hand_time_series.png)

# %% slideshow={"slide_type": "skip"}
plt.plot(IndShockSimExample.history['cNrm'][150:200,0:5]) #individual consumption paths for first 5
plt.xlabel('Time')
plt.ylabel('Individual consumption path')
plt.title('individual consumption paths', fontsize=12)
plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Model simulation
# * simulates the consumption response to income under the two views
#     - The simulation environment is chosen to match the empirical environment very closely
# * the author simulates the consumption reaction of 200 individuals to the receipt of a tax refund every 12 months over a period of 4 years
#     - For each tax refund received, the author calculates the MPC and cash on hand of each individual

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Variable definitions
# * main variables used in the analysis are the MPC and cash on hand
#     - MPC at time $t$ for individual $i$
#         - $MPC_{i t}=\frac{\Delta C_{i t}}{\Delta Y_{i t}}=\frac{\sum_{j=t}^{t+2} c_{i j}-\sum_{j=t-1}^{t-3} c_{i j}}{refund_{it}}$
#     - Pre-refund cash on hand at time $t$ for individual $i$
#         - $coh_{i t}^{P R}=\frac{\sum_{j=t-1}^{t-3} x_{i j}}{3}$
#     - Average cash on hand for individual $i$
#         - $\overline{c o h}_{i}=\frac{\sum_{j=t}^{T} x_{i j}}{T}$

# %% [markdown] slideshow={"slide_type": "slide"}
# #### The relationship between MPC and cash on hand (original figure in paper)
# * under the circumstances view
# ![Figure04_Relationship_MPC_coh_under_circumstances_view.png](attachment:Figure04_Relationship_MPC_coh_under_circumstances_view.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# * under the characteristics view
# ![Figure05_Relationship_MPC_coh_under_characteristics_view.png](attachment:Figure05_Relationship_MPC_coh_under_characteristics_view.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Variance decomposition
# * a quantitative measure that decomposes the relative role of circumstances and characteristics in explaining the MPC variance
#     - $MPC_{i t}=\alpha+\underbrace{\gamma_{1} \times \overline{coh}_{i}}_{\text {characteristics}}+\underbrace{\gamma_{2} \times\left(coh_{i t}^{P R}-\overline{coh}_{i}^{P R}\right)}_{\text {circumstances}}+\varepsilon_{i t}\;$, where $\mathbb{E}(\epsilon_{it}) = 0$.
# * so the variance is
#    - $\operatorname{var}\left(MPC_{i t}\right)=\operatorname{var}(\alpha)+\operatorname{var}\left(\gamma_{1} \times \overline{\operatorname{coh}}_{i}\right)+\operatorname{var}\left(\gamma_{2} \times\left(\operatorname{coh}_{i t}^{P R}-\overline{\operatorname{coh}}_{i}^{P R}\right)\right)+\operatorname{var}\left(\varepsilon_{i t}\right)$
# * fraction explained by cash on hand that is attributable to characteristics
#     - $\operatorname{var}\left(\gamma_{1} \times \overline{\operatorname{coh}}_{i}\right)=\sigma_{\text {char }}^{2}$
#     - $\operatorname{var}\left(\gamma_{2} \times\left(\operatorname{coh}_{i t}^{P R}-\overline{\operatorname{coh}}_{i}^{P R}\right)\right)=\sigma_{\text {circ }}^{2}$
#     - $\phi_{\text {char }}=\frac{\sigma_{\text {char }}^{2}}{\sigma_{\text {char }}^{2}+\sigma_{\text {circ }}^{2}}$

# %% [markdown]
# #### HARK result

# %% [markdown]
# ##### Circumstances View (experimental code 2)

# %%
# simulation for distribution of cash on hand
# 200 consumers
IdiosyncDictCircumView={
    # Parameters shared with the perfect foresight model
    "CRRA": 1.00000000000001,                           # Coefficient of relative risk aversion
    "Rfree": 1 + 0.01/12,                         # Interest factor on assets
    "DiscFac": 0.9941,                       # Intertemporal discount factor, monthly frequency
    "LivPrb" : 12*[1.0],                     # Survival probability, 0.99979
    "PermGroFac" : [1.0, 1.0, 1.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Permanent income growth factor
    
    #Parameters that specify the income distribution over the lifecycle
    "PermShkStd" : 12*[0.000001],                  # Standard deviation of log permanent shocks to income
    "PermShkCount" : 7,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : 12*[0.2],                  # Standard deviation of log transitory shocks to income
    "TranShkCount" : 7,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.00,                     # Probability of unemployment while working
    "IncUnemp" : 0.3,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.000,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.0,                      # Flat income tax rate (legacy parameter, will be removed in future)

    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 20,                       # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 48,                     # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra" : [None],                 # Additional values to add to aXtraGrid

    # A few other paramaters
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool" : True,                    # Whether to calculate the value function during solution
    "CubicBool" : False,                   # Preference shocks currently only compatible with linear cFunc
    "T_cycle" : 12,                        # Number of periods in the cycle for this agent type
                                           # from example, T_cycle = 4 means quarterly data
                                           # https://hark.readthedocs.io/en/latest/example_notebooks/IndShockConsumerType.html?highlight=quarterly#

    # Parameters only used in simulation
    "AgentCount" : 200,                  # Number of agents of this type
    "T_sim" : 120+48,                          # Number of periods to simulate
    "aNrmInitMean" : -6.0,                 # Mean of log initial assets
    "aNrmInitStd"  : 1.0,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
}

# %%
# circumstances view; pooled cross section
# MPC

num_consumers = 200                 # num of simulations we want

circumstances_view_consumers = IndShockConsumerType(**IdiosyncDictCircumView)
circumstances_view_consumers.solve(verbose=False)

circumstances_view_consumers.track_vars = ['aNrm','mNrm','cNrm','pLvl', 'MPCnow']
circumstances_view_consumers.initialize_sim()
circumstances_view_consumers.simulate()

# %%
# plot MPC against cash on hand;
# discard first 120 periods
for i in range(num_consumers):
    plt.scatter(circumstances_view_consumers.history['mNrm'][120:,i], 
                circumstances_view_consumers.history['MPCnow'][120:,i], color = 'b', alpha=0.5)
    
plt.xlabel('Cash on hand')
plt.ylabel('MPC')

plt.title('Fig 4a pooled cross section', fontsize=12)
plt.xlim(0, 4)
plt.ylim(-0.2, 0.8)
plt.show()

# %%
# plot average MPC against average cash on hand;
for i in range(num_consumers):
    plt.scatter(np.average(circumstances_view_consumers.history['mNrm'][120:,i]), 
                np.average(circumstances_view_consumers.history['MPCnow'][120:,i]), color = 'b', alpha=0.5)
    
plt.xlabel('Average MPC')
plt.ylabel('Average cash on hand')
plt.xlim(0, 4)
plt.ylim(-0.1, 0.8)
plt.title('Fig 4b average', fontsize=12)
plt.show()

# %% [markdown]
# ##### Characteristics View (experimental code 2)

# %%
# simulation for distribution of cash on hand
# 200 consumers
IdiosyncDictCharView={
    # Parameters shared with the perfect foresight model
    "CRRA": 1.00000000000001,                           # Coefficient of relative risk aversion
    "Rfree": 1 + 0.01/12,                         # Interest factor on assets
    "DiscFac": 0.9941,                       # Intertemporal discount factor, monthly frequency
    "LivPrb" : 12*[1.0],                     # Survival probability, 0.99979
    "PermGroFac" : [1.0, 1.0, 1.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Permanent income growth factor
    
    #Parameters that specify the income distribution over the lifecycle
    "PermShkStd" : 12*[0.000001],                  # Standard deviation of log permanent shocks to income
    "PermShkCount" : 7,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : 12*[0.2],                  # Standard deviation of log transitory shocks to income
    "TranShkCount" : 7,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.00,                     # Probability of unemployment while working
    "IncUnemp" : 0.3,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.000,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.0,                      # Flat income tax rate (legacy parameter, will be removed in future)

    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 20,                       # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 48,                     # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra" : [None],                 # Additional values to add to aXtraGrid

    # A few other paramaters
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool" : True,                    # Whether to calculate the value function during solution
    "CubicBool" : False,                   # Preference shocks currently only compatible with linear cFunc
    "T_cycle" : 12,                        # Number of periods in the cycle for this agent type
                                           # from example, T_cycle = 4 means quarterly data
                                           # https://hark.readthedocs.io/en/latest/example_notebooks/IndShockConsumerType.html?highlight=quarterly#

    # Parameters only used in simulation
    "AgentCount" : 1,                  # Number of agents of this type
    "T_sim" : 120+48,                          # Number of periods to simulate
    "aNrmInitMean" : -6.0,                 # Mean of log initial assets
    "aNrmInitStd"  : 1.0,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
}

# %%
from HARK.distribution import Uniform

num_consumer_types = 200   # num of types we want
char_view_consumers = []   # initialize an empty list

charConsumer = IndShockConsumerType(**IdiosyncDictCharView)

discFacDispersion = 0.0190
bottomDiscFac     = 0.9941 - discFacDispersion
topDiscFac        = 0.9941 + discFacDispersion

# draw discFac from uniform distribution with U(0.9941 - \Delta, 0.9941 + \Delta)
DiscFac_list  = Uniform(bot=bottomDiscFac,top=topDiscFac,seed=606).approx(N=num_consumer_types).X

# now create types with different disc factors
for i in range(num_consumer_types):
    newConsumer = deepcopy(charConsumer)
    newConsumer.DiscFac    = DiscFac_list[i]
    newConsumer.AgentCount = 1
    newConsumer.T_sim      = 120+48
    newConsumer.update_income_process()
    newConsumer.solve()
    
    char_view_consumers.append(newConsumer)

# simulate and keep track mNrm and MPCnow
for i in range(num_consumer_types):
    char_view_consumers[i].track_vars = ['aNrm','mNrm','cNrm','pLvl', 'MPCnow']
    char_view_consumers[i].initialize_sim()
    char_view_consumers[i].simulate()

# %%
# plot MPC against cash on hand;
for i in range(num_consumers):
    plt.scatter(char_view_consumers[i].history['mNrm'], 
                char_view_consumers[i].history['MPCnow'], color = 'b', alpha=0.5)
    
plt.xlabel('Cash on hand')
plt.ylabel('MPC')

plt.xlim(0, 4)
plt.ylim(-0.2, 0.8)
plt.title('Fig 5a pooled cross section', fontsize=12)
plt.show()

# %%
# plot average MPC against average cash on hand;
for i in range(num_consumers):
    plt.scatter(np.average(char_view_consumers[i].history['mNrm']), 
                np.average(char_view_consumers[i].history['MPCnow']), color = 'b', alpha=0.5)
    
plt.xlabel('Average cash on hand')
plt.ylabel('Average MPC')

plt.xlim(0, 4)
plt.ylim(-0.1, 0.8)
plt.title('Fig 5b average', fontsize=12)
plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ![Table06_Variance_decomp.png](attachment:Table06_Variance_decomp.png)

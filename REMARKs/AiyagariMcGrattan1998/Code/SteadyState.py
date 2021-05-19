import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType

from HARK.distribution import DiscreteDistribution
from HARK.utilities import plot_funcs

from time import process_time
from copy import deepcopy, copy

mystr = lambda number: "{:.4f}".format(number)
do_simulation = True

"""
# Transition matrix calculated using Julia

P = zeros(N, N);
for j = 1:N
    μ = ρ*ys[j]
    for l = 1:N
        lowX = ys[l] - μ - step
        l == 1 && (lowX = ys[l] - μ - step - 100.0)
        highX = ys[l] - μ + step
        l == N && (highX = ys[l] - μ + step + 100.0)
        (val,err) = hcubature(x -> exp(-.5*x'*ivΣ*x)[1], lowX, highX,
        reltol=1e-4, abstol=0, maxevals=0)
        P[j,l] = (val ./ sqrt(2 *pi * Σ))
    end
end
 
P = P ./ sum(P, 2)
transition = P
"""

transition = np.array([[1.90787000e-01, 4.55383000e-01, 3.01749000e-01, 5.00608633e-02,
        2.00160000e-03, 1.84984000e-05, 3.82913000e-08],
       [5.20813000e-02, 3.01749000e-01, 4.55383000e-01, 1.73993422e-01,
        1.64242000e-02, 3.67205000e-04, 1.87299000e-06],
       [8.77448000e-03, 1.21520000e-01, 4.19444000e-01, 3.65695768e-01,
        8.02333000e-02, 4.27914000e-03, 5.33123000e-05],
       [8.89025000e-04, 2.95073000e-02, 2.35589000e-01, 4.68029350e-01,
        2.35589000e-01, 2.95073000e-02, 8.89025000e-04],
       [5.33123000e-05, 4.27914000e-03, 8.02333000e-02, 3.65695768e-01,
        4.19444000e-01, 1.21520000e-01, 8.77448000e-03],
       [1.87299000e-06, 3.67205000e-04, 1.64242000e-02, 1.73993422e-01,
        4.55383000e-01, 3.01749000e-01, 5.20813000e-02],
       [3.82913000e-08, 1.84984000e-05, 2.00160000e-03, 5.00608633e-02,
        3.01749000e-01, 4.55383000e-01, 1.90787000e-01]])

# Create Dictionary

init_perfect_foresight = {
    'CRRA': 2.0,          # Coefficient of relative risk aversion,
    'Rfree': 1.03,        # Interest factor on assets
    'DiscFac': 0.991,      # Intertemporal discount factor
    'LivPrb': [0.98],     # Survival probability
    'PermGroFac': [1.01],  # Permanent income growth factor
    'BoroCnstArt': None,  # Artificial borrowing constraint
    'MaxKinks': 400,      # Maximum number of grid points to allow in cFunc (should be large)
    'AgentCount': 10000,  # Number of agents of this type (only matters for simulation)
    'aNrmInitMean': 0.0,  # Mean of log initial assets (only matters for simulation)
    'aNrmInitStd': 1.0,  # Standard deviation of log initial assets (only for simulation)
    'pLvlInitMean': 0.0,  # Mean of log initial permanent income (only matters for simulation)
    # Standard deviation of log initial permanent income (only matters for simulation)
    'pLvlInitStd': 0.0,
    # Aggregate permanent income growth factor: portion of PermGroFac attributable to aggregate productivity growth (only matters for simulation)
    'PermGroFacAgg': 1.0,
    'T_age': None,       # Age after which simulated agents are automatically killed
    'T_cycle': 1         # Number of periods in the cycle for this agent type
}


init_idiosyncratic_shocks = dict(
    init_perfect_foresight,
    **{
        # assets above grid parameters
        "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
        "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
        "aXtraNestFac": 3,  # Exponential nesting factor when constructing "assets above minimum" grid
        "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
        "aXtraExtra": [
            None
        ],  # Some other value of "assets above minimum" to add to the grid, not used
        # Income process variables
        "PermShkStd": [0.1],  # Standard deviation of log permanent income shocks
        "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
        "TranShkStd": [0.1],  # Standard deviation of log transitory income shocks
        "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
        "UnempPrb": 0.05,  # Probability of unemployment while working
        "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
        "IncUnemp": 0.3,  # Unemployment benefits replacement rate
        "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
        "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
        "tax_rate": 0.0,  # Flat income tax rate
        "T_retire": 0,  # Period of retirement (0 --> no retirement)
        "vFuncBool": False,  # Whether to calculate the value function during solution
        "CubicBool": False,  # Use cubic spline interpolation when True, linear interpolation when False
    }
)

init_aiyagari = copy(init_idiosyncratic_shocks)
init_aiyagari["MrkvArray"] = [transition]
init_aiyagari["UnempPrb"] = 0  # to make income distribution when employed
init_aiyagari["global_markov"] = False
aiyagari_consumer = MarkovConsumerType(**init_aiyagari)
aiyagari_consumer.cycles = 0
aiyagari_consumer.vFuncBool = False  # for easy toggling here

income_dist1 = DiscreteDistribution(np.array([1.0]), [np.array([1]), np.array([0.01])])  
income_dist2 = DiscreteDistribution(np.array([1.0]), [np.array([1]), np.array([0.03])])   
income_dist3 = DiscreteDistribution(np.array([1.0]), [np.array([1]), np.array([0.05])])   
income_dist4 = DiscreteDistribution(np.array([1.0]), [np.array([1]), np.array([0.07])]) 
income_dist5 = DiscreteDistribution(np.array([1.0]), [np.array([1]), np.array([0.09])])    
income_dist6 = DiscreteDistribution(np.array([1.0]), [np.array([1]), np.array([0.11])]) 
income_dist7 = DiscreteDistribution(np.array([1.0]), [np.array([1]), np.array([0.13])])     

aiyagari_consumer.IncShkDstn = [  
    [
        income_dist1,
        income_dist2,
        income_dist3,
        income_dist4,
        income_dist5,
        income_dist6,
        income_dist7,
    ]  
]

# Interest factor, permanent growth rates, and survival probabilities 
aiyagari_consumer.Rfree = np.array(7 * [aiyagari_consumer.Rfree])
aiyagari_consumer.PermGroFac = [
    np.array(7 * aiyagari_consumer.PermGroFac)
]
aiyagari_consumer.LivPrb = [aiyagari_consumer.LivPrb * np.ones(7)]
aiyagari_consumer.MrkvPrbsInit = [1/14, 2/14, 3/14, 4/14, 2/14, 1/7, 1/7]

aiyagari_consumer.solve()

# Preparation for simulation
L = 1
alpha = .36
beta =  1- alpha

aiyagari_consumer.cycles = 0
aiyagari_consumer.T_sim = 100
aiyagari_consumer.AgentCount = 10000
aiyagari_consumer.track_vars = ['aNrm','mNrm','cNrm','pLvl', 'aLvl']


time_pref = (1/aiyagari_consumer.DiscFac) - 1

tolerance = 10e-6
go=True
completed_loops = 0

r_1 = 1.03 - 1

# Simulation to solve for steady state
aiyagari_consumer.initialize_sim()
aiyagari_consumer.simulate()

AggA = np.mean(aiyagari_consumer.state_now['aLvl']) #Expectation of A

# r_2 = alpha * AggA ** (alpha - 1)
r_2 = alpha / (beta * AggA)

if r_2 > time_pref:
    
    r_2 = time_pref

while go:
    
   
    r_3 = (r_2 + r_1)/2
    
    aiyagari_consumer.Rfree = np.array(7 * [r_3]) + 1
    aiyagari_consumer.solve()
    aiyagari_consumer.initialize_sim()
    aiyagari_consumer.simulate()
    AggA = np.mean(aiyagari_consumer.state_now['aLvl']) 

    # K_3 = (r_3 / alpha) ** (1/(alpha-1))
    K_3 = alpha / (beta * r_3)
    
    
    distance= abs(AggA - K_3)

    
    completed_loops += 1
    
    print('Completed loops')
    print(completed_loops)
    print('distance')
    print(distance)
    
    go = distance >= tolerance and completed_loops < 15 
    
    if go == False:
        break
    
    if AggA < K_3:
        
        r_2 = r_3
    else:
        r_1 = r_3
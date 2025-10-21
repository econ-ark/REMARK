# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 00:51:38 2021

@author: William Du

python 3.8.8


"""
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from timeit import timeit
from HARK.distribution import DiscreteDistribution,combine_indep_dstns, Lognormal, MeanOneLogNormal, Uniform, calc_expectation
from HARK.utilities import get_percentiles, get_lorenz_shares, calc_subpop_avg
from HARK import Market, make_one_period_oo_solver

import HARK.ConsumptionSaving.ConsIndShockModel as ConsIndShockModel
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockSolver,
    IndShockConsumerType,
    PerfForesightConsumerType,
)
from HARK.distribution import Uniform
from HARK.ConsumptionSaving.ConsAggShockModel import CobbDouglasEconomy, AggShockConsumerType
from HARK import MetricObject, Market, AgentType
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt
from scipy.io import loadmat

from HARK.utilities import plot_funcs_der, plot_funcs


##############################################################################



##############################################################################
##############################################################################




    





##############################################################################


class FBSNK_agent(IndShockConsumerType):
    
   
    time_inv_ = IndShockConsumerType.time_inv_  + ["mu_u",
                                                   "L",
                                                   "SSPmu",
                                                   "wage",
                                                   "N",
                                                   "B",
                                                 
                                                   "s",
                                                   "dx",
                                                   "T_sim",
                                                   "jac",
                                                   "jacW",
                                                   "PermShkStd",
                                                   "Ghost",
                                                   
                                                    "PermShkCount",
                                                    "TranShkCount",
                                                    "TranShkStd",
                                                    "tax_rate",
                                                    "UnempPrb",
                                                    "IncUnemp",
                                                    "G",
                                                                               
                    
                                                  ]
    
    

    
    def __init__(self, cycles= 200, **kwds):
        
        IndShockConsumerType.__init__(self, cycles = 200, **kwds)
        
     
    
    

    def  update_income_process(self):
        
        self.wage = 1/(self.SSPmu) #calculate SS wage
        self.N = (self.mu_u*(self.IncUnemp*self.UnempPrb ) + self.G )/ (self.wage*self.tax_rate)#calculate SS labor supply from Budget Constraint
        

        
        
        
        PermShkDstn_U = Lognormal(np.log(self.mu_u) - (self.L*(self.PermShkStd[0])**2)/2 , self.L*self.PermShkStd[0] , 123).approx(self.PermShkCount) #Permanent Shock Distribution faced when unemployed
        PermShkDstn_E = MeanOneLogNormal( self.PermShkStd[0] , 123).approx(self.PermShkCount) #Permanent Shock Distribution faced when employed
        
        
        pmf_P = np.concatenate(((1-self.UnempPrb)*PermShkDstn_E.pmf ,self.UnempPrb*PermShkDstn_U.pmf)) 
        X_P = np.concatenate((PermShkDstn_E.X, PermShkDstn_U.X))
        PermShkDstn = [DiscreteDistribution(pmf_P, X_P)]
        self.PermShkDstn = PermShkDstn 
        
        TranShkDstn_E = MeanOneLogNormal( self.TranShkStd[0],123).approx(self.TranShkCount)#Transitory Shock Distribution faced when employed
        TranShkDstn_E.X = (TranShkDstn_E.X *(1-self.tax_rate)*self.wage*self.N)/(1-self.UnempPrb)**2 #NEED TO FIX THIS SQUARE TERM #add wage, tax rate and labor supply
        
        lng = len(TranShkDstn_E.X )
        TranShkDstn_U = DiscreteDistribution(np.ones(lng)/lng, self.IncUnemp*np.ones(lng)) #Transitory Shock Distribution faced when unemployed
        
        IncShkDstn_E = combine_indep_dstns(PermShkDstn_E, TranShkDstn_E) # Income Distribution faced when Employed
        IncShkDstn_U = combine_indep_dstns(PermShkDstn_U,TranShkDstn_U) # Income Distribution faced when Unemployed
        
        #Combine Outcomes of both distributions
        X_0 = np.concatenate((IncShkDstn_E.X[0],IncShkDstn_U.X[0]))
        X_1=np.concatenate((IncShkDstn_E.X[1],IncShkDstn_U.X[1]))
        X_I = [X_0,X_1] #discrete distribution takes in a list of arrays
        
        #Combine pmf Arrays
        pmf_I = np.concatenate(((1-self.UnempPrb)*IncShkDstn_E.pmf, self.UnempPrb*IncShkDstn_U.pmf))
        
        IncShkDstn = [DiscreteDistribution(pmf_I, X_I)]
        self.IncShkDstnN = IncShkDstn

        self.IncShkDstn = IncShkDstn
        self.add_to_time_vary('IncShkDstn')
        
        
        
        
        PermShkDstn_Uw = Lognormal(np.log(self.mu_u) - (self.L*(self.PermShkStd[0])**2)/2 , self.L*self.PermShkStd[0] , 123).approx(self.PermShkCount) #Permanent Shock Distribution faced when unemployed
        PermShkDstn_Ew = MeanOneLogNormal( self.PermShkStd[0] , 123).approx(self.PermShkCount) #Permanent Shock Distribution faced when employed
        
        TranShkDstn_Ew = MeanOneLogNormal( self.TranShkStd[0],123).approx(self.TranShkCount)#Transitory Shock Distribution faced when employed
        TranShkDstn_Ew.X = (TranShkDstn_Ew.X *(1-self.tax_rate)*(self.wage+self.dx)*self.N)/(1-self.UnempPrb)**2 #add wage, tax rate and labor supply
        
        lng = len(TranShkDstn_Ew.X )
        TranShkDstn_Uw = DiscreteDistribution(np.ones(lng)/lng, self.IncUnemp*np.ones(lng)) #Transitory Shock Distribution faced when unemployed
        
        IncShkDstn_Ew = combine_indep_dstns(PermShkDstn_Ew, TranShkDstn_Ew) # Income Distribution faced when Employed
        IncShkDstn_Uw = combine_indep_dstns(PermShkDstn_Uw,TranShkDstn_Uw)  # Income Distribution faced when Unemployed
        
        #Combine Outcomes of both distributions
        X_0 = np.concatenate((IncShkDstn_Ew.X[0],IncShkDstn_Uw.X[0]))
        X_1=np.concatenate((IncShkDstn_Ew.X[1],IncShkDstn_Uw.X[1]))
        X_I = [X_0,X_1] #discrete distribution takes in a list of arrays
        
        #Combine pmf Arrays
        pmf_I = np.concatenate(((1-self.UnempPrb)*IncShkDstn_Ew.pmf, self.UnempPrb*IncShkDstn_Uw.pmf))
        
        IncShkDstnW = [DiscreteDistribution(pmf_I, X_I)]
        
        self.IncShkDstnW = IncShkDstnW
        self.add_to_time_vary('IncShkDstnW')
        
 
    
    def get_Rfree(self):
        """
        Returns an array of size self.AgentCount with self.Rfree in every entry.
        Parameters
        ----------
        None
        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
    
        
        if self.jac==True or self.Ghost == True:
            RfreeNow = self.Rfree[self.t_sim - 1]* np.ones(self.AgentCount)
        else:
           
            RfreeNow = 1.05**.25 * np.ones(self.AgentCount)
            
        return RfreeNow
    
    def transition(self):
        
        
        pLvlPrev = self.state_prev['pLvl']
        aNrmPrev = self.state_prev['aNrm']
        RfreeNow = self.get_Rfree()

        # Calculate new states: normalized market resources and permanent income level
        pLvlNow = pLvlPrev*self.shocks['PermShk']  # Updated permanent income level
        # Updated aggregate permanent productivity level
        PlvlAggNow = self.state_prev['PlvlAgg']*self.PermShkAggNow
        # "Effective" interest factor on normalized assets
        ReffNow = RfreeNow/self.shocks['PermShk']
        bNrmNow = ReffNow*aNrmPrev         # Bank balances before labor income
        mNrmNow = bNrmNow + self.shocks['TranShk']  # Market resources after income
        
        
        if self.jac == True or self.jacW == True or self.Ghost==True:
        
            if self.t_sim == 0:
                
                for i in range(num_consumer_types):
                    if  self.DiscFac == consumers_ss[i].DiscFac:

                        mNrmNow = consumers_ss[i].history['mNrm'][self.T_sim-1,:]
                        pLvlNow = consumers_ss[i].history['pLvl'][self.T_sim-1,:]
                        print(self.DiscFac)
    

        return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None
    
    
    
    
FBSDict={
    # Parameters shared with the perfect foresight model
    "CRRA":2,                           # Coefficient of relative risk aversion
    "Rfree": 1.05**.25,                       # Interest factor on assets
    "DiscFac": 0.978,                     # Intertemporal discount factor
    "LivPrb" : [.98],  #.9725                  # Survival probability
    "PermGroFac" :[1.00],                 # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" :  [(0.01*4/11)**0.5],    # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [.2],        # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05,                     # Probability of unemployment while working
    "IncUnemp" : 0.08,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.1,                      # Flat income tax rate (legacy parameter, will be removed in future)

    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 20,                       # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 48,                     # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra" : [None],                 # Additional values to add to aXtraGrid

    # A few other parameters
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool" : True,                    # Whether to calculate the value function during solution
    "CubicBool" : False,                   # Preference shocks currently only compatible with linear cFunc
    "T_cycle" : 1,                         # Number of periods in the cycle for this agent type

    # Parameters only used in simulation
    "AgentCount" : 150000,                  # Number of agents of this type
    "T_sim" : 500,                         # Number of periods to simulate
    "aNrmInitMean" : np.log(.27)-(.5**2)/2,                 # Mean of log initial assets
    "aNrmInitStd"  : 0.3,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
    # new parameters
     "mu_u"       : .9 ,
     "L"          : 1.1, 
     "s"          : 1,
     "dx"         : .1,                  #Deviation from steady state
     "jac"        : True,
     "jacW"       : True, 
     "Ghost"      : False, 
     
    #New Economy Parameters
     "SSWmu " : 1.025 ,                      # Wage Markup from sequence space jacobian appendix
     "SSPmu" :  1.025,                       # Price Markup from sequence space jacobian appendix
     "calvo price stickiness":  .926,      # Auclert et al 2020
     "calvo wage stickiness": .899,        # Auclert et al 2020
     "B" : 0,                               # Net Bond Supply
     "G" : .01
     }


    
###############################################################################



###############################################################################

ss_agent = FBSNK_agent(**FBSDict)
ss_agent.cycles = 0
ss_agent.jac = False
ss_agent.jacW = False
ss_agent.dx = 0
ss_agent.T_sim = 1400
ss_agent.track_vars = ['aNrm','mNrm','cNrm','pLvl']

target = .27704816263570453

NumAgents = 150000

tolerance = .001

completed_loops=0

go = True

num_consumer_types = 7     # number of types 


'''
example = FBSNK_agent(**IdiosyncDict)
example.cycles = 0
example.jac = False
example.jacW = False
example.dx = 0
example.T_sim = 1400
example.track_vars = ['aNrm','mNrm','cNrm','pLvl']
example.DiscFac = .986

example.solve()
example.initialize_sim()
example.simulate()

'''

#center = .977 for Rfree 1.04**.25 and target 0.34505832912738216
center =.9681
#center =.7

while go:
    
    discFacDispersion = 0.0049
    bottomDiscFac     = center - discFacDispersion
    topDiscFac        = center + discFacDispersion
    
    #tail_N = 3
    #param_dist = Lognormal(mu=np.log(center)-0.5*spread**2,sigma=spread,tail_N=tail_N,tail_bound=[0.0,0.9], tail_order=np.e).approx(N=param_count-tail_N)
    
    #DiscFac_dist =Lognormal(mu=np.log(center)-0.5*discFacDispersion**2,sigma=discFacDispersion).approx(N=num_consumer_types-3,tail_N=3, tail_bound=[0,0.9])
    DiscFac_dist  = Uniform(bot=bottomDiscFac,top=topDiscFac,seed=606).approx(N=num_consumer_types)
    DiscFac_list  = DiscFac_dist.X
    
    consumers_ss = [] 
    
    # now create types with different disc factors
    for i in range(num_consumer_types):
        consumers_ss.append(deepcopy(ss_agent))
        
    for i in range(num_consumer_types):
        consumers_ss[i].DiscFac    = DiscFac_list[i]
        consumers_ss[i].AgentCount = int(NumAgents*DiscFac_dist.pmf[i])
    
    

    list_pLvl=[]
    list_aNrm =[]
    list_aLvl=[]
    litc=[]
    # simulate and keep track mNrm and MPCnow
    for i in range(num_consumer_types):
        consumers_ss[i].solve()
        consumers_ss[i].initialize_sim()
        consumers_ss[i].simulate()
        
        list_pLvl.append(consumers_ss[i].state_now['pLvl'])
        list_aNrm.append(consumers_ss[i].state_now['aNrm'])
        litc.append((consumers_ss[i].state_now['mNrm'] - consumers_ss[i].state_now['aNrm'])*consumers_ss[i].state_now['pLvl'])
        list_aLvl.append(consumers_ss[i].state_now['aLvl'])
        
        print('one consumer solved and simulated')
    
    pLvl = np.concatenate(list_pLvl)
    aNrm = np.concatenate(list_aNrm)
    c = np.concatenate(litc)
    a = np.concatenate(list_aLvl)
    AggA = np.mean(np.array(a))
    AggC = np.mean(np.array(c))

    
    
    if AggA - target > 0 :
        
       center = center - .0001
        
    elif AggA - target < 0: 
        center = center + .0001
        
    else:
        break
    
    
    print('Assets')
    print(AggA)
    print('consumption')
    print(AggC)
    print('center')
    print(center)
    
    distance = abs(AggA - target) 
    
    completed_loops += 1
    
    print('Completed loops')
    print(completed_loops)
    
    go = distance >= tolerance and completed_loops < 1
        
print("Done Computing Steady State")


###############################################################################
###############################################################################



funcs=[]
list_mLvl = []
list_mNrm = []
list_aNrm = []
for i in range(num_consumer_types):
    list_mLvl.append(consumers_ss[i].state_now['mNrm']*consumers_ss[i].state_now['pLvl'] )
    list_mNrm.append(consumers_ss[i].state_now['mNrm'])
    list_aNrm.append(consumers_ss[i].state_now['aNrm'])
    funcs.append(consumers_ss[i].solution[0].cFunc)

mNrm = np.concatenate(list_mNrm)   
mLvl = np.concatenate(list_mLvl)
aNrm = np.concatenate(list_aNrm)



x = np.linspace(0, 1.4, 1000, endpoint=True)

y=[]
for i in range(num_consumer_types):
    y.append(funcs[i](x))


h = np.histogram(mNrm, bins=np.linspace(0,1.4,num=1000))

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlabel('Cash on Hand')
plt.xlim(0,1.4)
ax1.plot(x, y[0], 'k' )

ax1.plot(x, y[1], 'm' )
ax1.plot(x, y[2], 'darkorange' )
ax1.plot(x, y[3], 'forestgreen' )
ax1.plot(x, y[4], 'deepskyblue' )
ax1.plot(x, y[5], 'r' )
ax1.plot(x, y[6], 'darkslategrey' )


ax1.set_ylim((0,.23))
ax1.set_ylabel('Consumption', color='k')


ax2= ax1.twinx()
ax2.hist(mNrm, bins=np.linspace(0,1.4,num=1000),color = 'darkviolet')
#ax2.hist(example.state_now['mNrm'],bins=np.linspace(0,1.4,num=1000),color= 'orange' )
ax2.set_ylim((0,1600))
ax2.set_ylabel('Number of Households', color='k')
#plt.savefig("Presentation.png", dpi=150)



################################################################################
################################################################################


class FBSNK2(FBSNK_agent):
    
    
    
     def update_solution_terminal(self):
        """
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.
        Parameters
        ----------
        none
        Returns
        -------
        none
        """
        
        for i in range(num_consumer_types):
            if self.DiscFac == DiscFac_list[i]:
                self.solution_terminal.cFunc = deepcopy(consumers_ss[i].solution[0].cFunc)
                self.solution_terminal.vFunc = deepcopy(consumers_ss[i].solution[0].vFunc)
                self.solution_terminal.vPfunc = deepcopy(consumers_ss[i].solution[0].vPfunc)
                self.solution_terminal.vPPfunc =  deepcopy(consumers_ss[i].solution[0].vPPfunc)
        

###############################################################################

params = deepcopy(FBSDict)


params['T_cycle']= 200
params['LivPrb']= params['T_cycle']*[.98]
params['PermGroFac']=params['T_cycle']*[1]
params['PermShkStd'] = params['T_cycle']*[(0.01*4/11)**0.5]
params['TranShkStd']= params['T_cycle']*[.2]
listRfree = params['T_cycle']*[ss_agent.Rfree]
params['Rfree'] = listRfree

###############################################################################


ghost_agent = FBSNK2(**params)
ghost_agent.pseudo_terminal = False
ghost_agent.IncShkDstn = params['T_cycle']*ghost_agent.IncShkDstn
ghost_agent.del_from_time_inv('Rfree')
ghost_agent.add_to_time_vary('Rfree')

ghost_agent.T_sim = params['T_cycle']
ghost_agent.cycles = 1
ghost_agent.track_vars = ['aNrm','mNrm','cNrm','pLvl','aLvl']
ghost_agent.dx = 0
ghost_agent.jac = False
ghost_agent.jacW = False
ghost_agent.Ghost = True


ghosts= [] 

for i in range(num_consumer_types):
    ghosts.append(deepcopy(ghost_agent))

# now create types with different disc factors
for i in range(num_consumer_types):
        ghosts[i].DiscFac   = DiscFac_list[i]
        ghosts[i].AgentCount = int(NumAgents*DiscFac_dist.pmf[i])
        #ghosts[i].solution_terminal = deepcopy(consumers_ss[i].solution[0]) ### Should it have a Deepcopy?
        #ghosts[i].cFunc_terminal_ = deepcopy(consumers_ss[i].solution[0].cFunc)

        
##############################################################################
     
     
listA_g = []
listH_Ag= []
listC_g = []
listH_g = []
listH_Mg =[]
listM_g = []
    
for k in range(num_consumer_types):
    ghosts[k].solve()
    ghosts[k].initialize_sim()
    ghosts[k].simulate()

    listH_g.append([ghosts[k].history['cNrm'], ghosts[k].history['pLvl']])
    listH_Ag.append(ghosts[k].history['aLvl'])
    listH_Mg.append(ghosts[k].history['mNrm'])
    
for j in range(ghost_agent.T_sim):

    litc_g=[]
    lita_g=[]
    litm_g=[]
    for n in range(num_consumer_types):
        litc_g.append(listH_g[n][0][j,:]*listH_g[n][1][j,:])
        lita_g.append(listH_Ag[n][j,:])
        litm_g.append(listH_Mg[n][j,:]*listH_g[n][1][j,:])
        
    Ag=np.concatenate(lita_g)
    Ag=np.mean(np.array(Ag))
    
    Cg = np.concatenate(litc_g)
    Cg = np.mean(np.array(Cg))

    Mg = np.concatenate(litm_g)
    Mg = np.mean(np.array(Mg))

    listM_g.append(Mg)
    listA_g.append(Ag)
    listC_g.append(Cg)
        
M_dx0 = np.array(listM_g)
A_dx0 = np.array(listA_g)
C_dx0 = np.array(listC_g)

plt.plot(C_dx0, label = 'Consumption Steady State')
plt.legend()
plt.show()


###############################################################################
###############################################################################


jac_agent = FBSNK2(**params)
jac_agent.pseudo_terminal = False
jac_agent.dx = 0.1
jac_agent.jac = True
jac_agent.jacW = False
jac_agent.IncShkDstn = params['T_cycle']*jac_agent.IncShkDstn
jac_agent.del_from_time_inv('Rfree')
jac_agent.add_to_time_vary('Rfree')
jac_agent.T_sim = params['T_cycle']
jac_agent.cycles = 1
jac_agent.track_vars = ['aNrm','mNrm','cNrm','pLvl','aLvl']


consumers = [] 

# now create types with different disc factors

for i in range(num_consumer_types):
    consumers.append(deepcopy(jac_agent))


for i in range(num_consumer_types):
        consumers[i].DiscFac    = DiscFac_list[i]
        consumers[i].AgentCount = int(NumAgents*DiscFac_dist.pmf[i])
        #consumers[i].solution_terminal = deepcopy(consumers_ss[i].solution[0]) ### Should it have a Deepcopy?
        #consumers[i].cFunc_terminal_ = deepcopy(consumers_ss[i].solution[0].cFunc)


###############################################################################
###############################################################################


testSet= [1,15,40,100]

Mega_list =[]
CHist = []
AHist = []
MHist = []
#for i in range(jac_agent.T_sim):
for i in testSet:
        
        listH_C = []
        listH_A = []
        listH_M = []
        listC = []
        listA = []
        listM = []
        for k in range(num_consumer_types):
            
            consumers[k].s = i 
            #consumers[k].IncShkDstn = (i-1)*ss_agent.IncShkDstn + jac_agent.IncShkDstnW + (params['T_cycle'] - i)*ss_agent.IncShkDstn
            consumers[k].Rfree = (i- 1)*[ss_agent.Rfree] + [ss_agent.Rfree + jac_agent.dx] + (params['T_cycle']- i)*[ss_agent.Rfree]


            consumers[k].solve()
            consumers[k].initialize_sim()
            consumers[k].simulate()
            
            listH_C.append([consumers[k].history['cNrm'],consumers[k].history['pLvl']])
            listH_M.append(consumers[k].history['mNrm'])
            #listA.append([consumers[k].history['aNrm'],consumers[k].history['pLvl']])
            listH_A.append(consumers[k].history['aLvl'])

            
        for j in range(jac_agent.T_sim):
            
            litc_jac= []
            lita_jac =[]
            litm_jac =[]
            for n in range(num_consumer_types):
                litc_jac.append(listH_C[n][0][j,:]*listH_C[n][1][j,:])
                lita_jac.append(listH_A[n][j,:])
                litm_jac.append(listH_M[n][j,:]*listH_C[n][1][j,:])

            
            c = np.concatenate(litc_jac)
            c = np.mean(np.array(c))
            listC.append(c)
            
            a = np.concatenate(lita_jac)
            a = np.mean(np.array(a))
            listA.append(a)
            
            
            m = np.concatenate(litm_jac)
            m = np.mean(np.array(m))
            listM.append(m)
            

        
        AHist.append(np.array(listA))
        MHist.append(np.array(listM))
        CHist.append(np.array(listC))
        #Mega_list.append(np.array(listC)- C_dx0)  # Elements of this list are arrays. The index of the element +1 represents the 
                                                  # Derivative with respect to a shock to the interest rate in period s.
                                                  # The ith element of the arrays in this list is the time t deviation in consumption to a shock in the interest rate in period s
        print(i)

###############################################################################
###############################################################################




'''
plt.plot(M_dx0, label = 'm steady state')
plt.plot(MHist[1], label = '15')
plt.plot(MHist[3], label = '100')
plt.plot(MHist[2], label = '40')
plt.legend()
plt.show()

plt.plot(MHist[1] - M_dx0, label = '15')
plt.plot(MHist[3] - M_dx0, label = '100')
plt.plot(MHist[2] - M_dx0, label = '40')
plt.plot(np.zeros(jac_agent.T_sim), 'k')
plt.legend()
plt.show()
'''


plt.plot(A_dx0, label = 'Asset steady state')
plt.plot(AHist[2], label = '40')
plt.plot(AHist[3], label = '100')
plt.plot(AHist[0], label = '15')
plt.plot(AHist[0], label = '1')
plt.ylim([.240,.33])
plt.xlabel("Period")
plt.ylabel("Aggregate Assets")
plt.title("Aggregate Assets")
plt.legend()
plt.show()



plt.plot((AHist[0][1:]- A_dx0[1:])/(jac_agent.dx), label = '1')
plt.plot((AHist[1][1:]- A_dx0[1:])/(jac_agent.dx), label = '15')
plt.plot((AHist[2][1:] - A_dx0[1:])/(jac_agent.dx), label = '40')
plt.plot((AHist[3][1:] - A_dx0[1:])/(jac_agent.dx), label = '100')
plt.plot(np.zeros(jac_agent.T_sim), 'k')
plt.xlabel("Period")
plt.ylabel("dA / dr")
plt.ylim([-.2,.6])
plt.title("Asset Jacobians")
plt.legend()
plt.show()
#plt.savefig("Rfree_jacobian.jpg", dpi=150)


plt.plot(C_dx0 , label = 'Steady State')
plt.plot(CHist[1], label = '15')
plt.plot(CHist[3], label = '100')
plt.plot(CHist[2], label = '40')
plt.title("Aggregate Consumption")
plt.ylabel("Aggregate Consumption")
plt.xlabel("Period")

plt.ylim([0.12,.1350])
plt.legend()
#plt.savefig("May3rdWage.jpg", dpi=150)
plt.show()





plt.plot((CHist[0][1:]- C_dx0[1:])/(jac_agent.dx), label = '1')
plt.plot((CHist[3][1:]- C_dx0[1:])/(jac_agent.dx), label = '100')
plt.plot((CHist[1][1:]- C_dx0[1:])/(jac_agent.dx), label = '15')
plt.plot((CHist[2][1:] - C_dx0[1:])/(jac_agent.dx), label = '40')
plt.plot(np.zeros(jac_agent.T_sim), 'k')
plt.ylim([-.04,.04])
plt.ylabel("dC / dr")

plt.xlabel("Period")
plt.title("Consumption Jacobians")
plt.legend()
#plt.savefig("wage_jacobian.jpg", dpi=150)
plt.show()


# =============================================================================
# 
# G=.01
# t=.1
# Inc = .08
# mho=.05
# 
# w = (1/1.025)
# N = (.9*(Inc*mho)+G)/ (w*t) 
# r = (1.05)**.25 -1
# print(N)
# 
# N1 = (.9*(.08*.05)+.01)/ (w*t) 
# 
# 
# q = ((1-w)*N)/r
# 
# print(N)
# print(N1-N)
# print(q)
# 
# =============================================================================

'''
funcs=[]
list_mLvl = []
list_mNrm = []
for i in range(num_consumer_types):
    list_mLvl.append(consumers_ss[i].state_now['mNrm']*consumers_ss[i].state_now['pLvl'] )
    list_mNrm.append(consumers_ss[i].state_now['mNrm'])
    funcs.append(consumers_ss[i].solution[0].cFunc)

mNrm = np.concatenate(list_mNrm)   
mLvl = np.concatenate(list_mLvl)
plot_funcs(funcs,0,1.4)
plt.hist(mNrm, bins=np.linspace(0,1.4,num=1000))
plt.show()

plt.hist(mLvl, bins=np.linspace(0,1.2,num=1000))
plt.show()



x = np.linspace(0, 1.4, 1000, endpoint=True)

y=[]
for i in range(num_consumer_types):
    y.append(funcs[i](x))


h = np.histogram(mNrm, bins=np.linspace(0,1.4,num=1000))

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlabel('Cash on Hand')
ax1.plot(x, y[0], 'm' )


ax1.plot(x, y[1], 'k' )
ax1.plot(x, y[2], 'darkorange' )
ax1.plot(x, y[3], 'forestgreen' )
ax1.plot(x, y[4], 'deepskyblue' )
ax1.plot(x, y[5], 'r' )
ax1.plot(x, y[6], 'darkslategrey' )
ax1.set_ylim((0,.23))
ax1.set_ylabel('Consumption', color='k')


ax2= ax1.twinx()
ax2.hist(mNrm, bins=np.linspace(0,1.4,num=1000),color= 'darkviolet')
ax2.set_ylim((0,1600))
ax2.set_ylabel('Number of Households', color='k')
#plt.savefig("Presentation.png", dpi=150)


sigma=.3
mean=np.log(.4)-(sigma**2)/2

print(np.exp(mean + (sigma**2)/2))
print((np.exp(sigma)-1)*np.exp(2*mean+sigma**2))

'''











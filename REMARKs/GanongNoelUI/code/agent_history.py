# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:53:33 2017

@author: Xian_Work
"""
import numpy as np
import solve_search
from solve_search import search, search_cost
from model_plotting import opt_dict_csv_in, mk_mix_agent
import copy
#############################################

def rand_hist(t,n,seed=0):
    """Generates a TxN array of random values between 0 and 1"""
    np.random.seed(seed)
    out=np.random.uniform(size=(t+1,n))
    return out
    
class agent_history_endo_js:
    def __init__(self,param_dict, z_vals, rand_array, sep_rate = None,
                 ex_search=False,
                 n_agents = 1500, sim_periods = 240):
        """
        Args:
            param_dict:         Behavioural parameters
            z_vals:             Benefit schedule
            rand_array:         Random TxN array to determine job transitions
            sep_rate:           Exogenous job separation rate
            ex_search:          True if job-finding rate is exogenous
            ex_search_rate:     Tuple of (jf_rate, immediately pre-exhaust rate)
            n_agents:           Number of agents to simulate
            sim_periods:        Agent lifetime
        """
        print "Initialize parameters and benefit schedule"
        self.p_d=copy.deepcopy(param_dict)
        self.p_d["z_vals"]=z_vals
        self.z_vals=z_vals
        
        self.rand_array=rand_array
        self.sep_rate=copy.deepcopy(self.p_d['Pi_'][0][0][0][1])
        self.ex_search=ex_search
        self.n_agents=n_agents
        self.sim_periods=sim_periods
        self.p_d['T_solve']=sim_periods
        
        if self.ex_search==True:
            self.p_d['solve_search']= False
            self.p_d['solve_V'] = False
       
        
    def sim_hist(self, cons_func=None, val_f=None,
                  s_perturb =0,perturb_periods=[],
                  exo_exhaust=[], exo_ui = [],
                  exo_assets=np.nan):
        """Simulates employment and consumption history w/ endogenous search
        
        Args:
            cons_func:      Consumption function. If None, compute the optimal consumption
                            function from the stored parameters
            val_f:          Value function. If None, compute the optimal consumption
                            function from the stored parameters
            exo_exhuast:    Exogenously set all agents exhausted in these (0-indexed) periods
            exo_ui:         Exogenously set all agents in 1st ui period in these (0-indexed) periods
            search_perturb :How much to manually perturb search from optimal
            perturb_periods:Periods to perturb in
            exo_assets:     Exogenously set asset value in perturbed periods
            
        For each of the n agents, compute the employment and consumption
        histories. Employment evolves according to an exogenous separation rate,
        the transition matrix self.Pi, and rand_array. For each unemployed state
        lookup the jf_rate in Pi. If the jf_rate is less than the corresponding
        value for that agent-time in rand_array, the agent stays unemployed.
        
        If ex_search is False, the effort encoded in Pi is detemined by the
        optimal search effort function. Otherwise,it  is exogenously set as
        ex_search_rate[1] if the agent is about to exhaust, and ex_search_rate[0]
        otherwise. 
        
        Consumption is specified either by the user-input 
        consumption function or by the computed optimal consumption function
        """
        if cons_func is not None:
            self.cf=cons_func
            if val_f is not None:
                self.value_f=val_f
        else:
            agent=solve_search.agent_series(self.p_d)
            agent.solve_agent()
            if not np.isclose(self.p_d['beta_hyp'], 1.0):
                agent.solve_agent_hb()
            self.cf=agent.consumption_f
            self.value_f=agent.value_f
            
        cf = self.cf

        #Function to get optimum effort in each period
        if self.ex_search==False:
            def effort(emp_state, assets, period):
                t=self.sim_periods -(period+1) #number of periods from final period
                search = solve_search.search(
                        self.value_f, t=t, a=assets, j=emp_state,
                        k=self.p_d['k'], beta_hyp=self.p_d['beta_hyp'],
                        phi=self.p_d['phi'],beta_var=self.p_d['beta_var'],
                        len_z = len(self.z_vals))
                return search + (period in perturb_periods)*s_perturb
        
        elif self.ex_search==True:
            self.pi_ = self.p_d['Pi_'][0][0] # exo job-finding rate matrix
            
            def effort(emp_state, assets,period):
                if emp_state==0:
                    return 0
                elif period > self.sim_periods-1: #no search in last period
                    return 0
                else:
                    return self.pi_[emp_state][0]
                
        #Function to determine transition
        def new_emp_state(effort, rand_val,current_emp_state):
            """Determines the employment state in the subsequent period
            
            Compares the effort level to a value drawn from the TxN rand_odds
            array. For an employed agent, if the sep_rate is less than the value
            the agent remains employed. For an unemployed agent, if the effort is 
            higher than the value the agent transitions to employment.
            """
            
            if current_emp_state<0:
                print('emp_state must be >= 0')
            
            #Probability of becoming unemployed
            if current_emp_state==0:
                if rand_val<self.sep_rate:
                    new_state=1 
                else:
                    new_state=0
                
            #Probability of transitioning into employment
            elif current_emp_state >0:
                if rand_val>effort:
                    new_state=min(len(self.p_d['z_vals'])-1,current_emp_state+1)
                else:
                    new_state=0
                    
            return new_state
          
        ##################################################33
        #Simulation here
        #Parameter values for computation
        sim_periods=self.sim_periods
        n_agents=self.n_agents
        rand_array=self.rand_array
        z_vals=self.p_d['z_vals']
        a0=self.p_d['a0']
        R=self.p_d['R']
        
        #Initial employment,search effort,  income, consumption, and assets
        e_hist=np.full((sim_periods,n_agents),np.nan,int) #emp_hist
        s_hist=np.full((sim_periods,n_agents),np.nan) #search effort
        y_hist=np.full((sim_periods,n_agents),np.nan) #income process
        m_hist=np.full((sim_periods,n_agents),np.nan) #m=CoH=a+y
        a_hist=np.full((sim_periods,n_agents),np.nan) #a_t=m_t-c_t
        c_hist=np.full((sim_periods,n_agents),np.nan) #cons is a func of m=CoH

        #Emp_state, income, CoH, consumption, effort, and assets in all subsequent periods
        for t in range(0,sim_periods):
            
            #set assets and employment
            if t == 0:
                a_hist[0] = np.repeat(a0,n_agents)
                e_hist[0] = (rand_array[0] < 0.11) #Some share of agents are unemployed in t=0
            
            else:
                a_hist[t] = R*(m_hist[t-1]-c_hist[t-1])
                for i in range(n_agents):
                    #emp_state is a probabilistic func of effort and emp_hist in prev period
                    e_hist[t][i]=new_emp_state(s_hist[t-1][i],rand_array[t-1][i],e_hist[t-1][i])  
            
            #hard code assets and employment history for perturbation testing
            if t in perturb_periods and ~np.isnan(exo_assets):
                a_hist[t] = np.repeat(exo_assets,n_agents)            
            if t in exo_exhaust and t in exo_ui:
                raise ValueError('cannot set agents exhausted and receiving UI simultaneously!')
            if t in exo_exhaust:
                e_hist[t] = [len(self.p_d['z_vals'])-1] * n_agents
            if t in exo_ui:
                e_hist[t] = [1] * n_agents
                
            #compute income, compute cash-on-hand    
            y_hist[t]=z_vals[e_hist[t]]
            m_hist[t]=a_hist[t] + y_hist[t]
                
            #compute consumption and search effort
            for i in range(n_agents):
                c_hist[t][i]=cf[sim_periods-(t+1)][e_hist[t][i]](m_hist[t][i])                  
                s_hist[t][i]=effort(e_hist[t][i],R*(m_hist[t][i]-c_hist[t][i]),t)
            
        if s_perturb != 0: #cleanup
            for x in np.nditer(s_hist, op_flags=['readwrite']):
                if np.isclose(x, s_perturb):
                    x[...] = 0
        
        #add results to agent_history_endo_js object            
        self.s_hist=s_hist
        self.e_hist=e_hist
        self.y_hist=y_hist
        self.m_hist=m_hist
        self.c_hist=c_hist
        self.a_hist=a_hist
        
        #compute summary statistics
        self.avg_emp = np.mean(e_hist==0)
        self.avg_search = (np.mean(s_hist)) / (1-self.avg_emp)
        self.avg_assets = np.mean(self.a_hist)
        
    def calc_avg_drop(self):
        """Calculates average drop in consumption at unemployment onset"""
        count=0
        c_e_sum=0
        c_ue_sum=0
        for t in range(0,sim_periods-1):
            for i in range(n_agents):
                if e_hist[t+1][i]==1:
                    c_e_sum += c_hist[t][i]
                    c_ue_sum += c_hist[t+1][i]
                    count+=1
        self.avg_ue_drop=(c_e_sum - c_ue_sum)/count
            
    def calc_welfare(self, discounted=True):
        """Calculates the lifetime utility of consumption, the total
        cost of search effort"""
        
        #Discounting, and fucntions to calculate npv utility
        p_d=self.p_d
        if discounted:
            disc_fac = 1-p_d['beta_var']
        else:
            disc_fac = 0 
        
        def utility(c, rho = p_d['rho']):
            return c**(1-rho)/(1-rho)
        
        
        def npv(flow,r=disc_fac):
            sum = 0
            for i in range(flow.shape[1]):
                sum += np.npv(r,flow[:,i])
            return sum
    
        #Utility from consumption smoothing (net)              
        u_cons = npv(utility(self.c_hist))
        
        #Utility from consumption smoothing gains (unemployed)
        c_hist_ue = np.multiply(self.c_hist, self.e_hist>0)
        c_hist_ue[c_hist_ue==0] = 0.0001 #so that utility can be computed
        u_cons_ue = npv(utility(c_hist_ue))
                
        #Utility from consumption smoothing losses (employed)
        c_hist_emp = np.multiply(self.c_hist, self.e_hist==0)
        c_hist_emp[c_hist_emp==0] = 0.0001 #so that utility can be computed
        u_cons_emp = npv(utility(c_hist_emp))
        
        #Job search costs
        base_cost = np.multiply(self.s_hist,self.e_hist>0)
        u_search = npv(-solve_search.search_cost(base_cost, k = p_d['k'], 
                        phi = p_d['phi']))
        
        self.u_cons = u_cons
        self.u_search = u_search
        self.u_tot = u_cons + u_search
    
    def calc_ex(self, ue_inc):
        """Calculate the total ui expenditure and taxes
        
        Args:
            ue_inc: the base income when unemployed, not receiving benefits
        """
        z_vals=self.z_vals
        
        #Tax revenue
        tax_rev=np.mean(self.e_hist==0) * (1- z_vals[0])
        self.tax_rev = tax_rev
        
        #Iterate through every employment state except state 0 (employed)
        tot_cost=0
        tot_periods=0
        for i in range(1,len(z_vals)):
            num_periods = np.mean(self.e_hist==i)        
            cost_per_period = z_vals[i]-ue_inc
            cost= cost_per_period * num_periods
            tot_periods += num_periods
            tot_cost += cost
        if np.isclose(tot_periods+np.mean(self.e_hist==0), 1):
            self.ui_ex = tot_cost 
        else:
            warnings.warn('Num periods does not sum to 1!')

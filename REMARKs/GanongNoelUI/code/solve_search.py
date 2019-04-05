#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:37:15 2017

@author: peterganong
"""    
from scipy.interpolate import InterpolatedUnivariateSpline
param_path="../Parameters/params_ui.json"
import setup_estimation_parameters as param             # Import parameters 
import numpy as np
import solve_cons


def search(V, t=239, a = param.a0_data, j = 1, k = param.k, phi = param.phi, 
           len_z = param.z_vals.shape[0], beta_hyp = 1, beta_var=0.998,
           search_cap=0.8, verbose = False):
    """ 
    Computes optimal search using value functions and assets
    
    Arguments:
        V -- value functions
        t -- t is the number of periods before the final period. In final period, t=0.
        a -- is the agent's assets in period t+1
    """
    if j == 0:
        return 0.0
    phi = float(phi) #prevent integer division when evaluating jf_rate

    if t == 0:
        return 0
    j = min(j+1,len_z-1)
    vf_t=max(t-1,0) #get value functions in next period - fixes time indexing bug
    dV = beta_var*beta_hyp*(V[vf_t][0](a) - V[vf_t][j](a))
    
    if dV < 0:
        warnings.warn('At time ' + str(t) + ' and state ' + str(j) + 
                         ', returns to search are negative!')
        jf_rate = 0
    else:    
        jf_rate = (dV/k)**(1.0/phi)
    
    if jf_rate > search_cap:
        if verbose:
            warnings.warn('At time ' + str(t) + ' and state ' + str(j) + 
                          ' and asset ' + str(a) +
                ', job-finding rate = ' + str(jf_rate) + ' >=' + str(search_cap))
        jf_rate = search_cap
    if verbose:
        print ('At time ' + str(t) + ', state ' + str(j) + ', assets ' + 
               str(round(a,2)) + ' job-finding rate = ' + str(round(jf_rate,2)))
        
    return jf_rate

def search_cost(s, k = param.k, phi = param.phi):
    return k*s**(1+phi)/(1+phi)              
                   
class agent_series:
    def __init__(self,params_dict,cons_func = 0, V = 0):
        for key in params_dict:
            setattr(self,key,params_dict[key])
            
        if cons_func != 0:
            self.consumption_f = cons_func
        else: 
            self.consumption_f = []
        if V != 0:
            self.value_f = V
        else: 
            self.value_f = []
            
        #check inputs
        self.len_z = self.z_vals.shape[0]
        self.s_list = list(range(self.len_z)) #xxx convert all s_list calls over to self.
        
        if not all(item.shape == (self.a_size+1,self.len_z,self.len_z) for item in self.Pi_): 
            raise ValueError("shape(z_vals) != shape(Pi).")
    ###########################################################################
    #Helper Functions for solve_agent
    
    
    def setup_a_grid(self, Tminust):
        """
        Finds lower bound on assets and computes the asset grid for period Tminust
        
        Natural borrowing constraint is sum from s=0 to s=T-t-1 of 
        [(z_min/R)*(1/R)^s] which, taking the sum, is (z_min/R)*(1-(1/R)^(T-t))/(1-1/R)
        
        Returns a_lower_bound and A-length grid of asset values
        """
        len_z=self.len_z
        if self.constrained:
            if Tminust == 0:
                a_lower_bound = np.maximum((-1)*self.L_, (-1)*self.z_vals[len_z-1]/self.R)
            else:
                a_lower_bound = np.maximum((-1)*self.L_, (-1)*(self.z_vals[len_z-1]/self.R)*(1-(1/self.R)**Tminust)/(1-(1/self.R)))
        else:
            a_lower_bound = (-1)*(self.z_vals[len_z-1]/self.R)*(1-(1/self.R)**Tminust)/(1-(1/self.R))

        a_grid = solve_cons.setup_grids_expMult(ming=a_lower_bound, 
            maxg=param.a_max, ng=self.a_size, timestonest=param.exp_nest)
        
        return a_lower_bound, a_grid
    
            
    def utility(self, c, rho = None):
        if rho == None:
            rho = self.rho
        return( c**(1-rho)/(1-rho))
    def utilityP(self, c, gam):
        return( c**-gam )
    def uP_(self, c,z=None):
        if z==None:
            z=self.rho
        return self.utilityP(c,z)
    
  
    #Consumption for arbitrary t
    def gothicC_t(self, a_,consumption_f_, rho_, R_, disc_fac, z_vals_, pmf_, tp1): 
        #utility-prime(cons_tomorrow(cash_on_hand_tomorrow) for every emp state)
        g_array = np.array([self.uP_(consumption_f_[tp1][state](R_ * a_ + z_vals_[state])) 
            for state in self.s_list]) 
        #expected marginal utility
        emu = [np.dot(g_array, pmf_[state]) for state in self.s_list]
        #use the inverse Euler equation to calculate cons today
        c_t = (disc_fac * R_ * np.array(emu))**(-1.0/rho_)  
        return c_t.tolist()

    def solve_cons_backward(self, Tminust,a_lower_bound,a_grid,
                            hyperbolic=False):
        """
        Use next period assets and probabilities to compute optimal cons today.
        
        Args:
            Tminust:    Period as measured using periods from end of life
            a_grid:     Beginning-of-period assets at Tminust-1
            hyperbolic: Whether to use hyperbolic discounting, and which
                        consumption function to update
            
        Updates self.consumption_f or self.consumption_f_hb
        Returns a J x A grid of optimal consumption choices
        """
        #initialize list of cash-on-hand values
        cash_on_hand = [[a_lower_bound] for state in self.s_list]
        #consumption is zero when cash-on-hand is at the asset lower bound
        cons_list = [[0] for state in self.s_list]
        
        if hyperbolic == False:
            disc_fac = self.beta_var
        elif hyperbolic == True:
            disc_fac = self.beta_var * self.beta_hyp
        
        for a_index, a_val in enumerate(a_grid): 
            c = self.gothicC_t(a_=a_val, consumption_f_= self.consumption_f, 
                          rho_=self.rho, R_=self.R, disc_fac=disc_fac, 
                          z_vals_=self.z_vals, pmf_=self.Pi_[Tminust - 1][a_index], 
                          tp1=Tminust - 1) 
            m = c + a_val/self.R 
            [cons_list[s].append(c[s]) for s in self.s_list]
            [cash_on_hand[s].append(m[s]) for s in self.s_list]       
        
        if hyperbolic == False:
            self.consumption_f.append([InterpolatedUnivariateSpline(
                cash_on_hand[s], cons_list[s], k=self.spline_k) for s in self.s_list])
        elif hyperbolic == True:
            self.consumption_f_hb.append([InterpolatedUnivariateSpline(
                cash_on_hand[s], cons_list[s], k=self.spline_k) for s in self.s_list])

        return np.array(cons_list)

    def solve_search_backward(self, Tminust,a_grid, hyperbolic=False):
        """ 
        Use next period value function to compute search today
        
        Args:
            Tminust:    Period as measured using periods from end of life
            a_grid:     Beginning-of-period assets at Tminust-1
            
        Updates self.Pi_
        Returns a J x A grid of optimal search choices
        """
        len_z=self.len_z
        s_list = self.s_list
        jf_grid=[0]* (len(a_grid))
        
        if hyperbolic==False:
            beta_hyp = 1
        elif hyperbolic == True:
            beta_hyp = self.beta_hyp
        
        for a_index, a_val in enumerate(a_grid):
            jf_vec = np.zeros(len_z)
            Pi = np.zeros([len_z, len_z])
            for s in s_list:
                jf_vec[s] = search(self.value_f, t=Tminust, a=a_val, 
                                  j=s,k=self.k, phi=self.phi,
                                  len_z=self.z_vals.shape[0],
                                  beta_hyp=beta_hyp,
                                  beta_var=self.beta_var)
                #a is end of period assets in Tminust
                # the 'search' function uses the vf from the future period automatically
            jf_grid[a_index]=jf_vec
            
            #update Pi with probability of not finding a job while getting UI   
            Pi[1:len_z,1:len_z] = Pi[1:len_z,1:len_z] + np.diagflat(1-jf_vec[1:len_z-1],k=1)
            Pi[len_z-1,len_z-1] = 1-jf_vec[len_z-1] #fill in p(not find) in exhausted state
            Pi[0:len_z,0] = jf_vec #fill in p(find)
            Pi[0,0:2] = param.Pi[0,0:2] #exogenous separations when employed
            if not all(item == 1 for item in np.sum(param.Pi,axis=1)):
                raise ValueError("This transition matrix has a row that does not sum to 1!")
            self.Pi_[Tminust-1][a_index] = Pi 
            
        return np.transpose(jf_grid)
    
    #helper function to compute value_func
    def solve_value_func_backward(self, Tmt, a_grid,c_grid,jf_grid):
        """ 
        Use next period assets & probabilities and this period cons and search
        to compute value func today
        
        Args:
            Tminust:    Period as measured using periods from end of life
            a_grid:     Beginning-of-period assets at Tminust-1
            c_grid:     Cons in Tminust
            jf_grid:    Search in Tminust
            
        Updates self.Pi_
        Returns a J x A grid of optimal search choices
        """
        s_list=self.s_list
        V= [[] for s in s_list]
        
        for a_index, a_val in enumerate(a_grid):
            for s in s_list:
                gv_array =  lambda Tmt, a_: np.array([self.value_f[Tmt][s](a_) for s in s_list]) 
                ev_tp1 = np.dot(gv_array(Tmt-1,a_val), self.Pi_[Tmt-1][a_index][s])
                value_f= self.beta_var* ev_tp1 
                V[s].append(value_f)
        
        c_grid_len = self.Pi_.shape[1]+1
        V_grid = self.utility(c_grid[:,1:c_grid_len]) - search_cost(jf_grid, k=self.k, phi=self.phi) + np.array(V)
        
        a_grid_today = [(a_grid - self.z_vals[s] + c_grid[s,1:c_grid_len])/self.R for s in s_list]

        V_func=[InterpolatedUnivariateSpline(a_grid_today[s], V_grid[s], 
                                    k=self.spline_k) for s in s_list]
        self.value_f.append(V_func) 
    
   
    ###########################################################################            
    def solve_agent(self):

        print ("Solving for rho = " + str(self.rho) + ", beta = " + str(self.beta_var) 
            + ", horizon T = " + str(self.T_solve) + ", limit = " + str(self.L_))
        
        # ==============================================================================
        # ============== Solve Consumption_{T} and value_{T} ==========================
        # ==============================================================================        
        a_lower_bound, a_grid = self.setup_a_grid(0)
        
        self.consumption_f.append([InterpolatedUnivariateSpline(
            list(range(10)), list(range(10)), k=self.spline_k) for s in self.s_list]) 
        
        if self.solve_V:
            V_func = [InterpolatedUnivariateSpline(a_grid, 
                self.utility(self.R*a_grid + self.z_vals[s]), k=self.spline_k) for s in self.s_list]
            self.value_f.append(V_func)

        # ==============================================================================
        # ============== Solve Consumption_{t}  and Value_{t} for all other t ==========
        # ==============================================================================

        for Tminust in range(1,self.T_solve):  
            
            a_lower_bound, a_grid =self.setup_a_grid(Tminust-1)
            
            if self.solve_search:
                jf_grid = self.solve_search_backward(Tminust,a_grid)
            
            c_grid = self.solve_cons_backward(Tminust,a_lower_bound, a_grid)
            
            if self.solve_search and self.solve_V:  
                self.solve_value_func_backward(Tminust,a_grid,c_grid,jf_grid)
    
    def solve_agent_hb(self):
        # ==============================================================================
        # ============== Solve Consumption_{T} =========================================
        # ==============================================================================        
        self.consumption_f_hb = []
        a_lower_bound, a_grid = self.setup_a_grid(0)
        
        self.consumption_f_hb.append([InterpolatedUnivariateSpline(
            list(range(10)), list(range(10)), k=self.spline_k) for s in self.s_list]) 
        
        # ==============================================================================
        # ============== Solve Consumption_{t}  and Value_{t} for all other t ==========
        # ==============================================================================
        for Tminust in range(1,self.T_solve):  
            
            a_lower_bound, a_grid =self.setup_a_grid(Tminust-1)
            
            if self.solve_search:
                jf_grid = self.solve_search_backward(Tminust,a_grid, hyperbolic=True)
            
            c_grid = self.solve_cons_backward(Tminust,a_lower_bound, a_grid, hyperbolic=True)
        
        self.consumption_f = self.consumption_f_hb
            
                    
    
    def compute_series(self,verbose=False, sparse=False, cf_sp = 0, vf_sp = 0, 
                       exhaust_date = 9, e=None, T_series=None):
        """Computes the realized series from the agent's consumption and search
        function, using an employment history
        
        Uses the agent's consumption and search functions to compute the optimal
        consumption, search, assets, and CoH series. By default, the computed
        series are of length self.T_series (specified in init) and based on 
        employment history self.e (also specified in init).
        """
        if e==None:
            e=self.e
        if T_series==None:
            T_series=self.T_series
    
        self.a_save = np.zeros(T_series+1)  #asset_path over time
        self.m_save = np.zeros(T_series+1)  #cash-on-hand path over time
        self.c_save = np.zeros(T_series+1)  #consumption over time
        self.s_save = np.zeros(T_series+1)  #job search over time
    
        for t in range(0, T_series+1): 
            if t==0:
                #First period cash-on-hand is initial bank balances plus employed income
                self.a_save[0] = self.a0
                self.m_save[0] = self.a0 + self.z_vals[e[0]]
                
            else:
                self.a_save[t] = self.m_save[t-1] - self.c_save[t-1]
                self.m_save[t] = self.R * self.a_save[t] + self.z_vals[e[t]]   #m_{t} = R*a_t-1 + y_{t}
            
            if not sparse or t >= exhaust_date:
                self.c_save[t] = self.consumption_f[self.T_solve-1-t][e[t]](self.m_save[t])
            elif sparse and t < exhaust_date:
                self.c_save[t] = cf_sp[self.T_solve-t][e[t]](self.m_save[t])
            
    
            
            #job search
            a_tp1 =  self.m_save[t] - self.c_save[t]
            
            if self.solve_search: 
                self.s_save[t] = search(V=self.value_f,t=self.T_solve-1-t,j=e[t],
                           a=a_tp1, phi = self.phi, k = self.k, 
                           len_z = self.z_vals.shape[0], beta_hyp = self.beta_hyp,
                           beta_var=self.beta_var, verbose = self.verbose) 
                if sparse and t < exhaust_date:
                    self.s_save[t] = search(V=vf_sp,t=self.T_solve-1-t,j=e[t],
                               a=a_tp1, phi = self.phi, k = self.k, 
                               len_z = self.z_vals.shape[0], beta_hyp = self.beta_hyp, 
                               beta_var=self.beta_var, verbose = self.verbose)
            else: 
                self.s_save[t] = np.nan


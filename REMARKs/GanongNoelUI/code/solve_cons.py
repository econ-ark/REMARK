from __future__ import division
import numpy as np
import csv
import os                                                
import copy
import cProfile
import pandas as pd
import warnings
from scipy.interpolate import InterpolatedUnivariateSpline
import setup_estimation_parameters as param             # Import parameters 

#========================================================
# FUNCTION FOR CREATING THE ASSET GRID
#=======================================================

#From HACKUtilities.py
def setup_grids_expMult(ming, maxg, ng, timestonest=20, a_huge = param.a_huge):
    """
    timestonest: int
        the number of times to nest the exponentiation
    """

    i=1
    gMaxNested = maxg - ming;
    while i <= timestonest:
        gMaxNested = np.log(gMaxNested+1);
        i += 1

    index = gMaxNested/float(ng)

    point = gMaxNested
    points = np.empty(ng)
    for j in range(1, ng+1):
            points[ng-j] = np.exp(point) - 1
            point = point-index
            for i in range(2,timestonest+1):
                points[ng-j] = np.exp(points[ng-j]) - 1

    points += ming
    
    #Add a_huge
    j = points.searchsorted(a_huge)
    points = np.insert(points, j, a_huge)
    
    return(points)

a_grid_default = setup_grids_expMult(ming=0, maxg=param.a_max, ng=param.a_size, timestonest=param.exp_nest)

#=======================================================
# TIMER CODE
#=======================================================
def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func
        
#=======================================================
# FUNCTION FOR SOLVING THE CONSUMPTION PROBLEM
#=======================================================
def utilityP(c, gam):                                            #marginal utility (or utility prime)
    return( c**-gam )
    
def solve_consumption_problem(rho_=param.rho,beta_var=param.beta, R=param.R,
        constrained=param.constrained,
        spline_k=param.spline_k, T=param.TT, L_=param.L, Pi_=param.Pi, 
        verbose = True, z_vals = param.z_vals):

    ###############
    # DEFINE CONSUMPTION FUNCTIONS
    ###############
    s_list = list(range(z_vals.shape[0]))
    if not (z_vals.shape[0] == Pi_.shape[0] and z_vals.shape[0] == Pi_.shape[1]):
        warnings.warn("In setup_estimation_parameters.py:    len(z_vals) != len(Pi).")


    def utilityP(c, gam):                                            #marginal utility (or utility prime)
        return( c**-gam )
        
    uP  = lambda c, z=rho_: utilityP(c, z)                           #marginal utility, unclear why we did this twice, took from Nathan.
    
    def gothicC_Tm1(a_, rho_, uP_, R_,  beta_, z_vals_, pmf_):          #consumption T-1(assets,...,pmf=probability mass function)       
        g = lambda z: uP_(R_*a_ + z)                              #consume all assets since it is the last period
        V = [(beta_ * R_ * np.dot(g(z_vals_), pmf_[state]))**(-1.0/rho_) for state in s_list]
        return V                   

  
    def gothicC_t(a_,consumption_f_, rho_, uP_, R_, beta_, z_vals_, pmf_, tp1): #Consumption for arbitrary t
        g_array = np.array([uP_(consumption_f_[tp1][state](R_ * a_ + z_vals_[state])) for state in s_list])                    
        V = [(beta_ * R_ * np.dot(g_array, pmf_[state]))**(-1.0/rho_) for state in s_list] #
        return V


    # Initialize a consumption functions container:
    consumption_f = []
    # ==============================================================================
    # ============== Solve Consumption_{T-1} and Consumption_{T} ===================
    # ==============================================================================

    #Find a_lower_bound for T-1. For constrained case, take max of hard borrowing constraint or natural borrowing constraint (whichever binds)
    if constrained:
        a_lower_bound = np.maximum((-1)*L_, (-1)*z_vals[-1]/R)
    else:
        a_lower_bound = (-1)*z_vals[-1]/R
    
    #Initialize the lists we will fill in with the loop. Limiting consumption is zero as m approaches m lower bound.
    # This is same as a_lower_bound, because a_t = m_t -c_t, so if m_t = a_lower_bound, c_t must equal zero.
    cons_list = [[0] for state in s_list]
    cash_on_hand = [[a_lower_bound] for state in s_list]

    #Set up a_grid
    a_grid = setup_grids_expMult(ming=a_lower_bound, maxg=param.a_max, ng=param.a_size, timestonest=param.exp_nest)

    for a in a_grid:                                                    #This loop calculates the optimal consumption for employed and unemployed at time T-1      
        c = gothicC_Tm1(a_=a, rho_=rho_, uP_=uP, R_=R, beta_=beta_var, z_vals_=z_vals,pmf_=Pi_)                 
        m = c + a 
        [cons_list[s].append(c[s]) for s in s_list]
        [cash_on_hand[s].append(m[s]) for s in s_list]
        
    #From the optimal grid-mapping of cash-on-hand to consumption, create a function using a linear spline.
    consumption_f.append([InterpolatedUnivariateSpline(cash_on_hand[s], cash_on_hand[s], k=spline_k) for s in s_list])                                     #First row of entries in consumption_f is consumption function at time T for e and u1,u2
    consumption_f.append([InterpolatedUnivariateSpline(cash_on_hand[s], cons_list[s], k=spline_k) for s in s_list])                                   # Next row entries are consumption functions at time T-1 for e and u1,u2
    
    # ==============================================================================
    # ============== Solve Consumption_{t} for all other t =========================
    # ==============================================================================
        
        #for t in range(T-2, -1, -1):
    if verbose == True:
        print "Solving for rho = " + str(rho_) + ", beta = " + str(beta_var) + ", horizon T = " + str(T) + ", limit = " + str(L_)
    
    for Tminust in range(2,T+1):    #Note, range(2,T+1) returns (2,3,4....(T+1)-1), so up to T.
    
        #Find a_lower_bound for period t. For constrained case, take max of hard borrowing constraint or natural borrowing constraint (whichever binds).
        # Natural borrowing constraint is sum from s=0 to s=T-t-1 of [(z_min/R)*(1/R)^s] which, taking the sum, is (z_min/R)*(1-(1/R)^(T-t))/(1-1/R)
        if constrained:
            a_lower_bound = np.maximum((-1)*L_, (-1)*(z_vals[8]/R)*(1-(1/R)**Tminust)/(1-(1/R)))
        else:
            a_lower_bound = (-1)*(z_vals[8]/R)*(1-(1/R)**Tminust)/(1-(1/R))

        a_grid = setup_grids_expMult(ming=a_lower_bound, maxg=param.a_max, ng=param.a_size, timestonest=param.exp_nest)
        
        cons_list = [[0] for state in s_list]
        cash_on_hand = [[a_lower_bound] for state in s_list]
        
        for a in a_grid:
            #Now we use the generic C_t cons function, because we use the fact that we know consumption function next period from steps above.
            c = gothicC_t(a_=a, consumption_f_= consumption_f, rho_=rho_, uP_=uP, R_=R, beta_=beta_var, z_vals_=z_vals,
                                                                     pmf_=Pi_, tp1=Tminust - 1)    # t plus 1              
            m = c + a 
            [cons_list[s].append(c[s]) for s in s_list]
            [cash_on_hand[s].append(m[s]) for s in s_list]
    
        #Create functions from these matched x and y grids, as above.
        consumption_f.append([InterpolatedUnivariateSpline(cash_on_hand[s], cons_list[s], k=spline_k) for s in s_list])

    return consumption_f

#====================================================================
# COMPUTE VALUE FUNCTION
#=====================================================================

def utility(c, rho = param.rho):
    return( c**(1-rho)/(1-rho))

def solve_value_func(cf_,  pmf_ = param.Pi, L_ = param.L,    
    a_grid = a_grid_default, beta_ = param.beta, z_vals_ = param.z_vals, 
    R_ = param.R, T = param.TT, spline_k = param.spline_k):
    
    print "Solving value func"
    a_grid = setup_grids_expMult(ming=-L_, maxg=param.a_max, ng=param.a_size, timestonest=param.exp_nest)
    
    s_list = list(range(z_vals_.shape[0]))
    #solve for one period
    gv_array =  lambda Tmt, a_: np.array([V_tp1_[s](R_ * (a_- cf_[Tmt][s](a_)) + z_vals_[s]) for s in s_list]) 
    v =         lambda Tmt, e, a_: utility(cf_[Tmt+1][e](a_+ z_vals_[e])) + beta_* np.dot(gv_array(Tmt,a_), pmf_[e])
   
    V_last = [InterpolatedUnivariateSpline(a_grid, utility(a_grid), k=spline_k) for s in s_list]
    V_tp1_= copy.deepcopy(V_last)        
    
    #initialize value functions container
    value_f = [V_last]

    for Tminust in range(1,T+1):
        V = [[v(Tminust,s,a_grid[0])] for s in s_list]
        for a_ in a_grid[1:len(a_grid)]:
            [V[s].append(v(Tminust,s,a_)) for s in s_list]
        V_func = [InterpolatedUnivariateSpline(a_grid.tolist(), V[s], k=spline_k) for s in s_list]    
        V_tp1_ = V_func        
        value_f.append(V_func)
        
    return value_f

#====================================================================
# FUNCTION FOR COMPUTING PATH OF CONSUMPTION DURING UNEMPLOYMENT SPELL
#=====================================================================
def compute_series(cons_func = 0, a0 = param.a0_data, T_series=9, T_solve=param.TT, e=[0,0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8], 
                   beta_var=param.beta,
                   rho=param.rho, verbose=True, L_=param.L, 
                   constrained=param.constrained, Pi_=param.Pi,
                   z_vals = param.z_vals, R = param.R, Rbor = param.R):
    if verbose == True:
        print "Compute series starting with a0 = " + str(a0)
    if cons_func == 0:
        cons_func = solve_consumption_problem(rho_=rho, beta_var=beta_var, R=R,
               constrained=constrained, T=T_solve, L_=L_, Pi_=Pi_, 
              verbose = verbose, z_vals = z_vals)
         
    #Initiate vectors to fill in
    a_save = np.zeros(T_series+1)  #asset_path over time
    m_save = np.zeros(T_series+1)  #cash-on-hand path over time
    c_save = np.zeros(T_series+1)  #consumption over time
    
    #Set first period values
         #Note on transition equations:
         #a_t = assets at end of period t = m_t - c_t
         #m_{t+1} = beggining of period resources / cash-on-hand = R*a_t + y_{t+1}
    m_save[0] = a0 + z_vals[e[0]]             #First period cash-on-hand is initial bank balances plus employed income
    c_save[0] = cons_func[T_solve][e[0]](m_save[0]) #First period consumption, as function of cash-on-hand
                                               #0th entry is time 0, recall consu_function works backwards, so last entry (entry T) is t=0 (first period). Now working forwards
    a_save[0] = m_save[0] - c_save[0]      #Assets at end of first period
        
    #Remaining time
    for t in range(0,T_series):
        m_save[t+1] = (R*(a_save[t] >= 0) + Rbor*(a_save[t]  < 0)) * a_save[t] + z_vals[e[t+1]]   #m_{t+1} = R*a_t + y_{t+1}
        c_save[t+1] = cons_func[T_solve-(t+1)][e[t+1]](m_save[t+1])
        a_save[t+1] = m_save[t+1] - c_save[t+1] 
        
    return a_save, c_save

if __name__ == "__main__":
    cf= solve_consumption_problem()



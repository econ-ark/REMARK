#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:27:14 2018

@author: peterganong
"""


#Import python modules

from scipy.optimize import fmin, brute
import scipy
param_path="../Parameters/params_ui.json" 
execfile("prelim.py")
from model_plotting import norm , compute_dist, sim_plot, make_base_plot


### Estimated parameters for other models
models_params ={}
for path in ["../Parameters/model_params_main.json", "../Parameters/model_params_sec.json"]:
    f= open(path)
    models_params.update(json.load(f))


#Model Target
cons_target = norm(param.JPMC_cons_moments, param.plt_norm_index)
cons_se=param.JPMC_cons_SE
cons_variance = np.square(cons_se)
cons_vcv = np.asmatrix(np.diag(cons_variance))
cons_wmat = np.linalg.inv(cons_vcv)
        
#Paramters
e_hist = models_params['sparsity']['e_hist_tm5']

def solve_cons_sparse(final_z, return_series = False, cf = None, 
                      T_ben_p1 = 8, z_vals = param.z_vals, rho = param.rho,
                      e_hist =e_hist):
    '''
    Computes the optimal consumption function for a sparse agent
    
    For a given perceived post-exhaustion income, computes the optimal consumption
    function for each employment state at each period.
    
    Args:
        final_z:    Perceived exhaustion income
        cf:         Fully attentive consumption function
        T_ben_pi:   Index of exhausted state
        z_vals:     Actual income process
    '''
    #Perceived Income Process
    z_sparse = np.insert(z_vals[:T_ben_p1],T_ben_p1,final_z)
    if T_ben_p1 == 7:
        z_sparse = np.insert(z_sparse,T_ben_p1+1,final_z)
    #print z_sparse
    cf_sparse_piece = solve_cons.solve_consumption_problem(z_vals = z_sparse,L_=sparse_L,
                                                           rho_ = rho, Pi_=param.Pi,
                                                           beta_var=sparse_beta)
                                                           
    cf_sparse = [[0] * 9 for i in range(param.TT+1)]
    for i in range(param.TT+1):
        for j in range(T_ben_p1):
            cf_sparse[i][j] = cf_sparse_piece[i][j]
        for j in range(T_ben_p1,9):
            cf_sparse[i][j] = cf[i][j]
    if not return_series:
        return cf_sparse
    else:
        a_sparse, c_sparse = solve_cons.compute_series(cons_func = cf_sparse, z_vals = z_vals,
                                                      Pi_=param.Pi, T_series = 15, beta_var=sparse_beta,
                                                      e=e_hist)
        return c_sparse

def compute_series_kappa(cons_funcs, a0 = param.a0_data, T_series=15, T_solve=param.TT, 
                         e=e_hist, 
                   z_vals = param.z_vals, R = param.R, Rbor = param.R):
    
    '''
    Computes the path of consumption from a list of consumption functions
    
    Args:
        cons_funcs:     List of consumption functions. List should have 8 consumption functions,
                        one for each employment state.
        a0:             Initial assets
        T_series:       Length of series to generate
        T_solve:        Final index of each consumption function
        e:              Employment history
        z_vals:         Actual income process
    
    Returns:
        a_save:         asset history
        c_save:         consumption history
    '''
    a_save, m_save, c_save = np.zeros(T_series+1), np.zeros(T_series+1), np.zeros(T_series+1)
    m_save[0] = a0 + z_vals[e[0]]    
    c_save[0] = cons_funcs[e[0]][T_solve][e[0]](m_save[0])
    a_save[0] = m_save[0] - c_save[0]    
    for t in range(0,T_series):
        m_save[t+1] = (R*(a_save[t] >= 0) + Rbor*(a_save[t]  < 0)) * a_save[t] + z_vals[e[t+1]]   #m_{t+1} = R*a_t + y_{t+1}
        c_save[t+1] = cons_funcs[e[t+1]][T_solve-(t+1)][e[t+1]](m_save[t+1])
        a_save[t+1] = m_save[t+1] - c_save[t+1] 
    return a_save, c_save

#Optimal Attention Function
def m_gen (c_sparse_default, dc_dm_initial, kappa_const): 
    '''
    Computes optimal attention for an array of (sparse) consumption
    
    Args:
        c_sparse_default:   Sparse consumption series
        dc_dm_initial:      Vector of marginal change in consumption wrt attention corresponding
                            to c_sparse_default
        kappa_const:        Constant that affects cost of attention
        
    Returns:
        Array of optimal attention, same length as c_sparse_default
    
    '''
    #solve agent FOC
    response_vs_cost = np.array((dc_dm_initial ** 2) * (c_sparse_default[:9] ** 2) * (1 / kappa_const ** 2))
    #implement sparse attention operator
    m = np.vstack((1 - 1 / response_vs_cost, np.zeros(9))).max(axis=0)
    m[8] = 1 # at exhaustion income is perceived correctly
    return m


def check_kappa_consistent(m_init, kappa, cf=None, verbose=False):
    '''
    Returns the quadratic distance between cons_seed and cons_implied
    
    m_init is a seed value of attention that implies a seed value of perceived z_exhaust.
    This seed value of perceivied z_exhaust generates optimal consumption c_sparse_default
    
    c_sparse_default also implies an array of optimal attention over the UI period, m. m
    generates an array of perceived ui_exhaust and an array of consumption functions for 
    each period of UI. This generates a new optimal consumption series, c_sparse
    
    This function returns the absolute distance between c_sparse_default and c_sparse. If
    the distance is small, then the generated consumption series is consistent with kappa.
    
    Args:
        kappa:      Kappa to check
        m_init:     Seed for initial attention
        verbose:    Return the consumption series?
    
    '''
    print('Testing for kappa = {0:.2f}'.format(kappa))
    dm = 0.01
    z_ui = param.z_vals[1]
    z_exhaust = param.z_vals[-1]
    final_z_func = lambda m: z_ui - m * (z_ui - z_exhaust)

            
    #define a fixed point here as having an initial value for z_tilde which is similar to the value micro-founded on kappa
    #iterate to find a mutually compatible m and kappa
    c_sparse_default = solve_cons_sparse(final_z = final_z_func(m_init), return_series = True, cf = cf)
    c_sparse_dm = solve_cons_sparse(final_z = final_z_func(m_init) - (0.83-0.54)*dm, return_series = True, cf = cf)
    dc_dm = (c_sparse_default[3:3+9]-c_sparse_dm[3:3+9])/dm
    m = m_gen(c_sparse_default, dc_dm ,kappa)
    print(m)
    
    #Create a list of consumption functions - sparse consumption func for each t<8
    cf_sparse = [] #calculate consumption function
    for t, m_val in enumerate(m):
        cf_sparse.append(solve_cons_sparse(final_z = final_z_func(m_val), cf = cf))
        
    a_sparse, c_sparse = compute_series_kappa(cf_sparse) #compute consumption path
    
    abs_dist = np.sum(np.absolute(c_sparse - c_sparse_default))
    print('absolute_distance is {0:.4f}'.format(abs_dist))
    
    if verbose==True:
        return {'abs_dist':abs_dist,
                'c_sparse_default': c_sparse_default,
                'c_sparse':c_sparse,
                'cf_sparse':cf_sparse,
                'm':m,
                'm_init':m_init}
    else:
        return abs_dist


def find_consistent_cons(kappa, cf):
    '''
    Finds the consumption series that is consistent with some value of kappa
    '''
    m_init_consistent = scipy.optimize.minimize_scalar(check_kappa_consistent, bounds=(0, 1),
                                                       args=(kappa, cf, False),
                                                       tol=0.1,
                                                       method = 'bounded',
                                                       options={'maxiter':3})
    return check_kappa_consistent(m_init_consistent['x'], kappa, cf, verbose=True)
    


#### Production Plot #######
sparse_beta = models_params['sparsity']['beta_var']
sparse_L = models_params['sparsity']['L_']
sparse_kappa = models_params['sparsity']['kappa']
cf= solve_cons.solve_consumption_problem(L_=sparse_L, Pi_=param.Pi, beta_var=sparse_beta) 

cons_out  = find_consistent_cons(sparse_kappa, cf)
c_sparse_error = cons_out['abs_dist']
c_sparse = cons_out['c_sparse']
cf_sparse = cons_out['cf_sparse']
attn = cons_out['m']
c_sparse_norm = norm(c_sparse, param.plt_norm_index)

## Main Plots
cons_plot = make_base_plot(0,param.c_moments_len,
                           param.moments_len_diff, param.s_moments_len,
                           cons_target, [0]*param.c_moments_len,
                           cons_se,[1]*param.s_moments_len)
cons_plot.add_series('Model: Sparse Attention',
                     c_sparse_norm, np.array([1]*param.s_moments_len))
cons_plot.plot('sparsity', 'Spending in Data and Sparse Attention Model',
               None, 'ignore',
               cons_t0=-5, tminus5=True, GOF=0,
               cons_ylim=(0.65,1.03),
               cons_legend_loc = (0.25,0.22))

#Add representative agent
rep_agent_pd = {"a0": param.a0_data, "T_series":T_series, "T_solve":param.TT, 
               "e":param.e_extend, 
               "beta_var":param.beta, "beta_hyp": param.beta_hyp, "a_size": param.a_size,
               "rho":param.rho, "verbose":False, "L_":param.L, 
               "constrained":param.constrained, "Pi_":np.zeros((param.TT+1,param.a_size+1,9,9)),
               "z_vals" : param.z_vals, "R" : param.R, "Rbor" : param.R, 
               "phi": param.phi, "k":param.k, "spline_k":param.spline_k, "solve_V": True,
               "solve_search": True}
for t in range(param.TT+1):
    for a_index in range(param.a_size+1):
        rep_agent_pd['Pi_'][t][a_index] = param.Pi
rep_agent_pd['T_series']=len(rep_agent_pd['e'])-1

rep_agent_pd.update(models_params['est_params_1b1k'])

        
cons_plot.add_agent('Model: Representative Agent', rep_agent_pd['e'],
                    param.c_plt_start_index,
                    param.s_plt_start_index,
                    param.plt_norm_index,
                    *[(1,rep_agent_pd)], verbose=True)        
cons_plot.plot('sparsity_with_rep',
               '''Spending in Data, Sparse Attention Model,\nand Representative Agent Model''',
               None, 'ignore',
               cons_t0=-5, tminus5=True, GOF=0,
               cons_ylim=(0.65,1.03))
cons_plot.plot('sparsity_with_GOF',
               '''Spending in Data, Sparse Attention Model,\nand Representative Agent Model''',
               None, 'ignore',
               cons_t0=-5, tminus5=True, GOF=True,
               cons_ylim=(0.65,1.03),
               cons_legend_loc = (0.32,0.22))
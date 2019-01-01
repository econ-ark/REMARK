# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 09:08:21 2018

@author: Xian_Work
"""
import scipy
from scipy.optimize import fmin, brute
from model_plotting import norm , compute_dist,gen_evolve_share_series,  mk_mix_agent

#Import parameters and other UI modules
param_path="../Parameters/params_ui.json" 
execfile("prelim.py")

####################################
#Model Target and Base Plots
####################################
#Data Series to plot against
data_tminus5 = norm(param.JPMC_cons_moments, param.plt_norm_index)
data_tminus5_search = param.JPMC_search_moments
###Use vcv weights###
cons_se_vcv_tminus5 =param.JPMC_cons_SE
search_se_vcv = param.JPMC_search_SE

#Targets
opt_target_cons = data_tminus5
cons_wmat = np.linalg.inv(np.asmatrix(np.diag(np.square(cons_se_vcv_tminus5))))

opt_target_search = data_tminus5_search[param.moments_len_diff: param.moments_len_diff +  param.s_moments_len]
search_wmat = np.linalg.inv(np.asmatrix(np.diag(np.square(search_se_vcv))))


#########################################
# Baseline Parameters
#########################################
pd_base = {"a0": param.a0_data, "T_series":T_series, "T_solve":param.TT, 
           "e":param.e_extend, 
           "beta_var":param.beta, "beta_hyp": param.beta_hyp, "a_size": param.a_size,
           "rho":param.rho, "verbose":False, "L_":param.L, 
           "constrained":param.constrained, "Pi_":np.zeros((param.TT+1,param.a_size+1,9,9)),
           "z_vals" : param.z_vals, "R" : param.R, "Rbor" : param.R, 
           "phi": param.phi, "k":param.k, "spline_k":param.spline_k, "solve_V": True,
           "solve_search": True}

for t in range(param.TT+1):
    for a_index in range(param.a_size+1):
        pd_base['Pi_'][t][a_index] = param.Pi
pd_base['T_series']=len(pd_base['e'])-1

### Estimated standard model
f = open("../Parameters/model_params_main.json")
models_params = json.load(f)
f.close
est_params_1b1k = models_params['est_params_1b2k']
est_params_1b2k = models_params['est_params_1b2k']


pd_1b1k = copy.deepcopy(pd_base)
for k, v in est_params_1b1k.iteritems():
    if k in pd_base.keys():
        pd_1b1k.update({k:v})


pd_1b2k = copy.deepcopy(pd_base)
for k, v in est_params_1b2k.iteritems():
    if k in pd_base.keys():
        pd_1b2k.update({k:v})
        
weights_1b2k = (est_params_1b2k['w_lo_k'], 1- est_params_1b2k['w_lo_k'])
params_1b2k = ('k', )
vals_1b2k = ((est_params_1b2k['k0'], ),
             (est_params_1b2k['k1'], ),)

################################################
# Functions to estimate delta for a given gamma
###############################################
def gen_agent_bhvr(agent):
    c_start = param.c_plt_start_index
    s_start = param.s_plt_start_index
    
    series_dict = gen_evolve_share_series(pd_base['e'],
                                     c_start, s_start,
                                     len(opt_target_cons),
                                     param.plt_norm_index,
                                     *agent,
                                     verbose = True,
                                     normalize = True)
    cons_out = series_dict['w_cons_out']
    search_out = series_dict['w_search_out'][s_start-c_start : s_start-c_start+len(opt_target_search)]
    
    return {'cons_out':cons_out, 'search_out':search_out}




def find_opt_delta_1b2k(gamma, opt_type= None):
    """For a given gamma, finds the discount factor delta(beta_var)
    that generates behavior to best fit the data moments"""
        
    def obj_func(delta_in):
            delta = delta_in[0]
            #Generate agent
            if opt_type == "1b1k":
                pd_temp = copy.deepcopy(pd_1b1k)
                pd_temp.update({'rho':gamma})
                pd_temp.update({'beta_var':delta})
                
                agent = [(1,pd_temp)]
            
            elif opt_type == "1b2k":
                pd_temp = copy.deepcopy(pd_1b2k)
                pd_temp.update({'rho':gamma})
                pd_temp.update({'beta_var':delta})
            
                agent = mk_mix_agent(pd_temp, params_1b2k, vals_1b2k, weights_1b2k)
            
            #Compute predicted behavior
            series_out = gen_agent_bhvr(agent)
            cons_out = series_out['cons_out']
            search_out = series_out['search_out']
                    
            #Calculate distance from targets
            cons_dist = compute_dist(cons_out, opt_target_cons, cons_wmat)
            search_dist = compute_dist(search_out, opt_target_search, search_wmat)
            
            return cons_dist + search_dist     
        
    opt_out = scipy.optimize.minimize(obj_func, [pd_base['beta_var'],],
                                      bounds=[(0.9,1.0),],
                                      options = {'maxiter':15})
    return(opt_out)


###############################################
# Estimate best delta for a range of gamma vals
###############################################
gammas_out = {}

for gamma in [0.9999, 4.0, 10.0]:
    opt_out_1b1k = find_opt_delta_1b2k(gamma, opt_type ="1b1k")
    opt_out_1b2k = find_opt_delta_1b2k(gamma, opt_type ="1b2k")
    
    opt_delta_1b1k = opt_out_1b1k['x'][0] 
    opt_delta_1b2k = opt_out_1b2k['x'][0] 
    
    key_1b1k = "est_params_1b1k_fix_gamma_" + str(int(np.round(gamma)))
    key_1b2k = "est_params_1b2k_fix_gamma_" + str(int(np.round(gamma)))

    gammas_out.update({key_1b1k: {'beta_var':opt_delta_1b1k, 'rho':gamma}})
    gammas_out.update({key_1b2k: {'beta_var':opt_delta_1b2k, 'rho':gamma}})

with open('../Parameters/model_params_robust_gamma.json', 'w') as f:
    json.dump(gammas_out, f, indent=0)

    
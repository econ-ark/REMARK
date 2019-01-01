# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 08:48:22 2018

@author: Xian_Work
"""
import scipy
from scipy.optimize import fmin, brute
from model_plotting import norm , compute_dist,gen_evolve_share_series,  mk_mix_agent

#Import parameters and other UI modules
param_path="../Parameters/params_ui.json" 
execfile("prelim.py")

####################
###### Setup ######
####################
opt_process = "serial" #Runs only the minimum number of estimations required to replicate models in the paper. Other option is "parallel"
in_path = './est_models_in/'
out_path = './est_models_out/'
##############################


### Input arguments for optimization ###
init_conditions_list = []
final_conditions_dict = {}

if opt_process == "serial":
    #Creates a list of dicts; each dict corresponds to an optimization to perform
    #Later pipes all the optimization results to one json 
    master_infile = 'initial_conditions_master.json'
    master_outfile = '../../Parameters/model_params_main.json'
    
    with open(in_path+master_infile, "r") as read_file:
        init_conditions_master = json.load(read_file)
    for i in init_conditions_master:
        init_conditions_list.append(i)
    
elif opt_process == "parallel": 
    #Creates a length-1 list from a json, containing an optimization dict
    #Then deletes the input json
    #Later pipes output of optimization to a json in the output directory
    
    #If running parallel, this python script must be run once for each optimization;
    #grid_sims_helper.py contains func to convery csvs of initial conditions
    #to json input for this script, and func to convert json outputs of this
    #script into csvs
    in_path = './est_models_in/'
    out_path = './est_models_out/'
    
    def get_files(start_str = "sim_"):
        """Return files that start with start_str"""
        n = len(start_str)
        file_list = [f for f in os.listdir(in_path) if f[0:n] == start_str]
        return file_list
    
    #Get input dict and delete from input directory
    filename = get_files()[0]
    with open(in_path + filename, 'r') as f:
        opt_input = json.load(f)
    init_conditions_list.append(opt_input)
    os.remove(in_path + filename)

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

################################################
#Function to generate agent 
################################################
def gen_agent(vals_dict):
    #Optimizing the model with two search types and hyperbolic discounting
    
    #Common parameters to be updated in all models
    agent_pd = copy.deepcopy(pd_base)
    if 'phi' in vals_dict.keys():
        agent_pd.update({'phi':vals_dict['phi']})
                     
    #Representative Agent
    if vals_dict['opt_type'] == '1b1k':
        agent_pd.update({'beta_var':vals_dict['beta_var'],
                         'beta_hyp':vals_dict['beta_hyp'],
                         'L_':vals_dict['L_'],
                         'k':vals_dict['k']})
        agent = [(1, agent_pd),]
        return(agent)
    
    #Heterogeneity in Search Costs
    if vals_dict['opt_type'] == '1b2k':
        agent_pd.update({'beta_var':vals_dict['beta_var'],
                         'beta_hyp':vals_dict['beta_hyp'],
                         'L_':vals_dict['L_']})
    
        k0 = vals_dict['k0']
        k1 = vals_dict['k1']
        
        agent_weights = (vals_dict['w_lo_k'], 1- vals_dict['w_lo_k'])
        agent_params = ('k', )
        agent_vals = ((k0, ),
                      (k1, ),)
    
    #Heterogeneity in Consumption Preferences and in Search Costs    
    if vals_dict['opt_type'] == '2b2k' or vals_dict['opt_type'] == '2b2k_fix_xi' or vals_dict['opt_type'] == '2b2k_fix_b1':
        agent_pd.update({'beta_var':vals_dict['beta_var'],
                         'L_':vals_dict['L_']})
        #Heterogeneous k types
        k0 = vals_dict['k0']
        k1 = vals_dict['k1']
        #Heterogeneous beta types
        b0 = vals_dict['b0']
        b1 = vals_dict['b1']

        #Weights
        w_lo_k = vals_dict['w_lo_k']
        w_hi_k = 1 - w_lo_k
        w_lo_beta = vals_dict['w_lo_beta']
        w_hi_beta = 1 - w_lo_beta
        
        w_b0_k0 = w_lo_k * w_lo_beta
        w_b1_k0 = w_lo_k * w_hi_beta
        w_b0_k1 = w_hi_k * w_lo_beta
        w_b1_k1 = w_hi_k * w_hi_beta

        #Make agent
        agent_weights = (w_b0_k0, w_b1_k0, w_b0_k1, w_b1_k1)
        agent_params = ('beta_hyp', 'k' )
        agent_vals = ((b0, k0),
                      (b1, k0),
                      (b0, k1),
                      (b1, k1))
        
    ### Additional models - robustness checks to different types of heterogeneity
    if vals_dict['opt_type'] == '2d2k':
        agent_pd.update({'L_':vals_dict['L_']})
        agent_pd.update({'beta_hyp':1})
        
        #Heterogeneous k types
        k0 = vals_dict['k0']
        k1 = vals_dict['k1']
        #Heterogeneous beta types
        d0 = vals_dict['d0']
        d1 = vals_dict['d1']

        #Weights
        w_lo_k = vals_dict['w_lo_k']
        w_hi_k = 1 - w_lo_k
        w_lo_delta = vals_dict['w_lo_delta']
        w_hi_delta = 1 - w_lo_delta
        
        w_d0_k0 = w_lo_k * w_lo_delta
        w_d1_k0 = w_lo_k * w_hi_delta
        w_d0_k1 = w_hi_k * w_lo_delta
        w_d1_k1 = w_hi_k * w_hi_delta

        #Make agent
        agent_weights = (w_d0_k0, w_d1_k0, w_d0_k1, w_d1_k1)
        agent_params = ('beta_var', 'k' )
        agent_vals = ((d0, k0),
                      (d1, k0),
                      (d0, k1),
                      (d1, k1))

    agent = mk_mix_agent(agent_pd, agent_params, agent_vals, agent_weights)        
    return agent
################################################
#Function to generate Consumption and Search behaviour
################################################    
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


########################################
#Main estimation execution starts here
#######################################
for opt_input in init_conditions_list:
    ####################################
    #Baseline parameters
    ####################################
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
    
    if 'fix_gamma' in opt_input:
        pd_base.update({'rho':opt_input['fix_gamma']})
    if opt_input['opt_type'] == '2b2k_fix_xi':
        pd_base.update({'phi':opt_input['phi']})
        
    #######################################################
    #### Indexing of arguments that will be optimized over 
    #######################################################
    #Representative Agent
    if opt_input['opt_type'] == '1b1k':
        args_index = {'beta_var':0,
                      'beta_hyp':1,
                      'k':2,
                      'L_':3,
                      'phi':4}
        args_bounds = [(0.9, 1.0),
                       (0.2, 1.0),
                       (1.0, 300.0),
                       (0.0, 12.0),
                       (0.5, 2.0)]
    
    #Heterogeneity in Search Costs
    elif opt_input['opt_type'] == '1b2k':
        args_index = {'beta_var':0,
                      'beta_hyp':1,
                      'k0':2,
                      'k1':3,    
                      'L_':4,
                      'w_lo_k':5,
                      'phi':6}
        args_bounds = [(0.9, 1.0),
                       (0.2, 1.0),
                       (1.0, 300.0),
                       (1.0, 300.0),
                       (0.0, 12.0),
                       (0.001, 0.999),
                       (0.5, 2.0)]
    
    #Heterogeneity in Consumption Preferences and in Search Costs
    elif opt_input['opt_type'] == '2b2k':
        args_index = {'beta_var':0,
                      'L_':1,
                      'b0':2,
                      'b1':3,
                      'w_lo_beta':4,
                      'k0':5,
                      'k1':6,    
                      'w_lo_k':7,
                      'phi':8}
        args_bounds = [(0.9, 1.0),
                       (0.0, 12.0),
                       (0.2, 1.0),
                       (0.2, 1.0),
                       (0.001, 0.999),
                       (1.0, 300.0),
                       (1.0, 300.0),
                       (0.001, 0.999),
                       (0.5, 2.0)]
        
    elif opt_input['opt_type'] == '2b2k_fix_xi':
        args_index = {'beta_var':0,
                      'L_':1,
                      'b0':2,
                      'b1':3,
                      'w_lo_beta':4,
                      'k0':5,
                      'k1':6,    
                      'w_lo_k':7}
        args_bounds = [(0.8, 1.0),
                       (0.0, 12.0),
                       (0.1, 1.0),
                       (0.1, 1.0),
                       (0.001, 0.999),
                       (1.0, 300.0),
                       (1.0, 300.0),
                       (0.001, 0.999)]
        
    elif opt_input['opt_type'] == '2b2k_fix_b1':
        args_index = {'beta_var':0,
                      'L_':1,
                      'b0':2,
                      'w_lo_beta':3,
                      'k0':4,
                      'k1':5,    
                      'w_lo_k':6,
                      'phi':7}
        args_bounds = [(0.9, 1.0),
                       (0.0, 12.0),
                       (0.2, 1.0),
                       (0.001, 0.999),
                       (1.0, 300.0),
                       (1.0, 300.0),
                       (0.001, 0.999),
                       (1.001, 2.0)]

    #Heterogeneity in Delta    
    elif opt_input['opt_type'] == '2d2k':
        args_index = {'L_':0,
                      'd0':1,
                      'd1':2,
                      'w_lo_delta':3,
                      'k0':4,
                      'k1':5,    
                      'w_lo_k':6,
                      'phi':7}
        args_bounds = [(0.0, 12.0),
                       (0.2, 1.0),
                       (0.2, 1.0),
                       (0.001, 0.999),
                       (1.0, 300.0),
                       (1.0, 300.0),
                       (0.001, 0.999),
                       (0.5, 2.0)]        
    args_index_rv = {v: k for k, v in args_index.iteritems()} #For processing opt_args_in
    ####################################################
    #Objective Function 
    ####################################################    
    def obj_func(opt_args_in = [], opt_type = None, verbose = False):
        
        ###Generate agent ###
        vals_dict = {'opt_type':opt_type}
        for key, value in args_index.iteritems():
            vals_dict.update({key:opt_args_in[value]})
        if opt_input['opt_type'] == '2b2k_fix_b1'.decode('utf8'):
            print('adding fixed b1 value')
            vals_dict.update({'b1':opt_input['b1']})
        agent = gen_agent(vals_dict)
        
        #Generate consumption and search behaviour
        series_out = gen_agent_bhvr(agent)
        cons_out = series_out['cons_out']
        search_out = series_out['search_out']
        
        #Calculate distance from targets
        cons_dist = compute_dist(cons_out, opt_target_cons, cons_wmat)
        search_dist = compute_dist(search_out, opt_target_search, search_wmat)
        
        if verbose==True:
            return (cons_dist, search_dist, cons_dist + search_dist)
        return cons_dist + search_dist    
          
        
    ###########################
    #Optimization    
    ###########################
    opt_args_in = []
    for i in range(len(args_index_rv)):
        opt_args_in.append(opt_input[args_index_rv[i]])
    
    opt_out = scipy.optimize.minimize(obj_func, opt_args_in,
                                      args = (opt_input['opt_type'], False),
                                      bounds=args_bounds,
                                      options = {'maxiter':13, 'ftol':0.001})
    distances = obj_func(opt_out['x'], opt_type = opt_input['opt_type'], verbose=True)    
    
    
    ###########################
    #Write results
    ###########################
    opt_args_out = copy.deepcopy(opt_input)
    #Initial Params
    for key, val in opt_input.iteritems():
        init_key = "init_" + key
        opt_args_out.update({init_key: val})
    
    #Optimized Params
    for key, val in args_index.iteritems():
        opt_args_out.update({key:opt_out['x'][val]})
                    
    ###For robustness checks where we estimate with different risk aversion        
    if 'fix_gamma' in opt_input:
            opt_args_out.update({'rho':opt_input['fix_gamma']})
    
    #Optimization Statistics
    opt_args_out.update({'GOF': distances[2],
                         'GOF_cons': distances[0],
                         'GOF_search': distances[1],
                         'term_stat': opt_out['message'],
                         'num_iters': opt_out['nit']})
    ###Write output
    if opt_process == "parallel":
        with open(out_path+filename, 'w') as f:
            json.dump(opt_args_out, f)
    
    elif opt_process == "serial":
        key = "est_params_" + opt_args_out['opt_type']
        if 'fix_gamma' in opt_input:
            key = key + '_fix_gamma_' + str(int(np.round(opt_input['fix_gamma'],0)))
        if opt_args_out['opt_type'] == ['2b2k_fix_xi']:
            key = key + '_' + str(opt_input['phi'])
        if opt_args_out['opt_type'] == ['2b2k_fix_b1']:
            key = key + '_' + str(opt_input['b1'])
        final_conditions_dict.update({key: opt_args_out})
        
###Final dump when optimizing in serial mode
if opt_process == "serial":
    with open(out_path + master_outfile, 'w') as f:
            json.dump(final_conditions_dict, f, indent=0)

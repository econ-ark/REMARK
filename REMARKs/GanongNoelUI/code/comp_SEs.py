# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 16:04:44 2017

@author: Xian_Work
"""
param_path="../Parameters/params_ui.json" 
execfile("prelim.py")
from model_plotting import gen_evolve_share_series, mk_mix_agent


###########################################
####Calculate gradients for each moment####
###########################################
def calc_gradients(func,func_args,free_args,perturb=0.001):
    """
    Calculates the gradient of an M->N function at a specified point,
    building up by peturbing one parameter at a time to caluclate partials
    
    Args:
        func:       The function the generates the m_hat -must return an array
        func_args:  Dict of all arguments to the function (fixed and varying)
                    and the values the gradient is to be evaluated at
        free_args:  List of keys in func_args to allow to vary
        peturb:     Size of peturbation for numerical gradient calculation

    """
    #The j-th element in grad_dict is a 1-d vector: the derivative of the j-th moment at its realized value
    #The i-th element in the vector is the partial wrt the i-th parameter

    #The vector of function output (moments) before peturbations
    m_hat=func(func_args)
    
    #Initialize output dictionary
    grad_dict={}
    for j in range(len(m_hat)):
        grad_dict.update({j:{}}) #For each of j values, have a dict of gradients
        
    #Iterate through peturbations of different parameters
    for key in free_args:
        param_val=func_args[key]
        d_minus = param_val - perturb
        d_plus = param_val + perturb
        temp_in_minus = copy.deepcopy(func_args) 
        temp_in_plus = copy.deepcopy(func_args)
        temp_in_minus[key] = d_minus #replace with peturbed value
        temp_in_plus[key] = d_plus
        m_minus = func(temp_in_minus)
        m_plus = func(temp_in_plus)
        
        #Fill in the partial wrt this param, for each function value (moment)
        for j in range(len(m_hat)):
            vals = [m_minus[j], m_hat[j], m_plus[j]] #j-th output moment
            partial = np.gradient(vals, perturb)[1]#Take the middle value
            grad_dict[j].update({key:partial})
            
    return grad_dict

###########################################
####Functions to compute Standard Errors
###########################################
def calc_param_se(model_type, p_d, free_args,
                  w_mat, lambda_hat,
                  emp_hist, cons_start,search_start, periods, norm_time,
                  het_vals=None, het_weights=None):
    """
    Computes the parameter estimates' standard errors
    
    Uses the same method as in add_agent to generate the series, then compares
    the results to the target, and finally generates the standard errors
    Args:
        model_type:     Specify the model type. 
        p_d:            Base parameter dictionary with fixed/calibrated parameters
                        at their specified values
        free_args:      List of estimated parameters in p_d, het_vals, and het_weights
        w_mat:          Weight matrix
        lambda_hat:     Variance of moments

        emp_hist:       Employment history to generate consumption and search series
        con_start:      Point in emp_hist to start generating cons and search series
        search_start:   Before this index in search series, search is 0 (dummy values)
        periods:        Length of series to generate
        norm_time:      Index of consumption series for normalization
    """
    def grad_func(in_dict):
        het_base=copy.deepcopy(p_d)
        for key, val in in_dict.iteritems(): #Update shared params
            if key in het_base.keys():
                het_base[key]=val

        #Create mixed agent
        if model_type == "1b2k":
            weights = (in_dict['w_lo_k'], 1-in_dict['w_lo_k'])
            params = ('k', )
            vals = ((in_dict['k0'],),
                    (in_dict['k1'],))
        if model_type == "2b1k":
            weights = (in_dict['w_lo_beta'], 1-in_dict['w_lo_beta'])
            params = ('beta_hyp', )
            vals = ((in_dict['b0'],),
                    (in_dict['b1'],))
        if model_type == "2b2k" or model_type == "2b2k_fix_xi" or model_type == "2b2k_fix_b1":
            w_lo_k = in_dict['w_lo_k']
            w_hi_k = 1 - w_lo_k
            w_lo_beta = in_dict['w_lo_beta']
            w_hi_beta = 1 - w_lo_beta
            
            w_b0_k0 = w_lo_k * w_lo_beta
            w_b1_k0 = w_lo_k * w_hi_beta
            w_b0_k1 = w_hi_k * w_lo_beta
            w_b1_k1 = w_hi_k * w_hi_beta

            weights = (w_b0_k0, w_b1_k0, w_b0_k1, w_b1_k1)
            params = ('beta_hyp', "k")
            vals = ((in_dict['b0'], in_dict['k0']),
                    (in_dict['b1'], in_dict['k0']),
                    (in_dict['b0'], in_dict['k1']),
                    (in_dict['b1'], in_dict['k1']))
        if model_type == "2d2k":
            w_lo_k = in_dict['w_lo_k']
            w_hi_k = 1 - w_lo_k
            w_lo_delta = in_dict['w_lo_delta']
            w_hi_delta = 1 - w_lo_delta
            
            w_d0_k0 = w_lo_k * w_lo_delta
            w_d1_k0 = w_lo_k * w_hi_delta
            w_d0_k1 = w_hi_k * w_lo_delta
            w_d1_k1 = w_hi_k * w_hi_delta
    
            weights = (w_d0_k0, w_d1_k0, w_d0_k1, w_d1_k1)
            params = ('beta_var', "k")
            vals = ((in_dict['d0'], in_dict['k0']),
                    (in_dict['d1'], in_dict['k0']),
                    (in_dict['d0'], in_dict['k1']),
                    (in_dict['d1'], in_dict['k1']))
               
        #Create the agent:
        if model_type == "1b1k":
            agent = [(1, in_dict)]
        else:
            agent = mk_mix_agent(het_base, params, vals, weights)
        
        #Compute series:
        series = gen_evolve_share_series(emp_hist, cons_start, search_start,
                                         periods, norm_time, *agent,
                                         verbose=True, normalize=True, w_check=False)
        cons=series['w_cons_out']
        search=series['w_search_out'][search_start-cons_start:search_start-cons_start+11] #11 is number of search moments
        moments=np.append(cons,search)
        return moments

    #Compute the gradients
    all_params = copy.deepcopy(p_d)
    if model_type != "1b1k":
        all_params.update(het_weights)
        all_params.update(het_vals)
    grad_dict=calc_gradients(grad_func,all_params,free_args,perturb=0.001) #free_args lists the parameters free to vary
    m_hat = grad_func(all_params)
          
    #Index free parameters in the matrix of partials:
    param_index={}
    index=0
    for key in grad_dict[0].keys():
        param_index.update({index:key})
        index+=1
    #Create the g_hat matrix of partials with the order    
    g_hat=np.array(np.zeros(((len(grad_dict),len(param_index)))))
    for m in range(len(grad_dict)):
        for n in range(len(param_index)):
            g_hat[m][n]=grad_dict[m][param_index[n]]
            
    #Compute standard errors (matrix algebra)
    g_hat = np.asmatrix(g_hat)
    g_hat_t = np.matrix.transpose(g_hat)
    w_mat= np.asmatrix(w_mat)
    lambda_hat = np.asmatrix(lambda_hat)
    
    LHS=np.linalg.inv(np.dot(np.dot(g_hat_t,w_mat),g_hat))
    MID= np.dot(np.dot(np.dot(np.dot(g_hat_t, w_mat), lambda_hat), w_mat), g_hat) 
    RHS= LHS
    variances=LHS * MID * RHS
    variances = (variances/len(free_args))
    
    #variances is the variance-covariance matrix of the parameters
    #param_index tells us which parameter is in which position in the variances matrix. The (i,i)
    #element in variances is the variance of the ith parameter in param_index
    
    #cleanup naming for clarity:
    colnames = []
    for i in range(len(param_index)):
        colnames.append(param_index[i])
    g_hat = pd.DataFrame(g_hat)
    g_hat.columns=colnames
    variances = pd.DataFrame(variances)
    variances.columns = colnames
    
    return (grad_dict, g_hat, param_index, variances, LHS, MID, RHS, m_hat)

###########################################
#### Setup for computing SEs ##############
###########################################
###Default buffer-stock agent###
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

###Weighting matrix and variance of moments###
cons_se=np.array(param.JPMC_cons_SE)
search_se = np.array(param.JPMC_search_SE)
se_stitch=np.append(cons_se,search_se)     
var_stitch=np.square(se_stitch)
vcv=np.asmatrix(np.diag(var_stitch))

lambda_hat = vcv
w_mat=np.linalg.inv(vcv)

#################################################
###############Calculate Standard Errors ########
################For spender-saver################
f = open("../Parameters/model_params_main.json")
model_params_main = json.load(f)
f.close

f = open("../Parameters/model_params_sec.json")
model_params_sec = json.load(f)
f.close

models_params = model_params_main
models_params.update(model_params_sec)

###Buffer stock - representative agent
pd_rep=copy.deepcopy(pd_base)
est_params_1b1k = models_params['est_params_1b1k']
pd_rep.update(est_params_1b1k)
free_args_rep = ['beta_var', 'beta_hyp', 'k', 'L_', 'phi']

###2k types (Standard Model)
est_params_1b2k = models_params['est_params_1b2k']
het_1b2k = copy.deepcopy(pd_rep)
het_1b2k.update({'beta_var':est_params_1b2k['beta_var'],'beta_hyp':est_params_1b2k['beta_hyp'],
                 'L_':est_params_1b2k['L_'], 'constrained':True, 'phi':est_params_1b2k['phi'] })

het_weights_1b2k ={'w_lo_k':est_params_1b2k['w_lo_k']}
het_vals_1b2k =  {'k0':est_params_1b2k['k0'], 'k1':est_params_1b2k['k1']}
free_args_1b2k = ['beta_var', 'beta_hyp', 'L_', 'k0', 'k1', 'w_lo_k', 'phi']


# 2b2k
est_params_2b2k = models_params['est_params_2b2k']
het_2b2k = copy.deepcopy(pd_rep)
het_2b2k.update({'beta_var':est_params_2b2k['beta_var'], 'L_':est_params_2b2k['L_'],
                 'constrained':True, 'phi':est_params_2b2k['phi']})

    
het_weights_2b2k =  {'w_lo_beta':est_params_2b2k['w_lo_beta'], 'w_lo_k':est_params_2b2k['w_lo_k']} 
het_vals_2b2k=  {'b0':est_params_2b2k['b0'], 'b1':est_params_2b2k['b1'],
                  'k0':est_params_2b2k['k0'], 'k1':est_params_2b2k['k1']}
free_args_2b2k = ['beta_var', 'L_', 'k0', 'k1', 'b0', 'b1',
                  'w_lo_beta', 'w_lo_k', 'phi']

# 2b2k, fix xi=1.0
est_params_2b2k_fix_xi = models_params['est_params_2b2k_fix_xi']
het_2b2k_fix_xi = copy.deepcopy(pd_rep)
het_2b2k_fix_xi.update({'beta_var':est_params_2b2k_fix_xi['beta_var'], 'L_':est_params_2b2k_fix_xi['L_'],
                        'constrained':True, 'phi':est_params_2b2k_fix_xi['phi']})

    
het_weights_2b2k_fix_xi =  {'w_lo_beta':est_params_2b2k_fix_xi['w_lo_beta'],
                            'w_lo_k':est_params_2b2k_fix_xi['w_lo_k']} 
het_vals_2b2k_fix_xi=  {'b0':est_params_2b2k_fix_xi['b0'],
                        'b1':est_params_2b2k_fix_xi['b1'],
                        'k0':est_params_2b2k_fix_xi['k0'],
                        'k1':est_params_2b2k_fix_xi['k1']}
free_args_2b2k_fix_xi = ['beta_var', 'L_', 'k0', 'k1', 'b0', 'b1',
                         'w_lo_beta', 'w_lo_k']

# 2b2, fix b1=1.0
est_params_2b2k_fix_b1 = models_params['est_params_2b2k_fix_b1']
het_2b2k_fix_b1 = copy.deepcopy(pd_rep)
het_2b2k_fix_b1.update({'beta_var':est_params_2b2k_fix_b1['beta_var'], 'L_':est_params_2b2k_fix_b1['L_'],
                 'constrained':True, 'phi':est_params_2b2k_fix_b1['phi']})

    
het_weights_2b2k_fix_b1 =  {'w_lo_beta':est_params_2b2k_fix_b1['w_lo_beta'],
                     'w_lo_k':est_params_2b2k_fix_b1['w_lo_k']} 
het_vals_2b2k_fix_b1 =  {'b0':est_params_2b2k_fix_b1['b0'], 'b1':est_params_2b2k_fix_b1['b1'],
                  'k0':est_params_2b2k_fix_b1['k0'], 'k1':est_params_2b2k_fix_b1['k1']}
free_args_2b2k_fix_b1 = ['beta_var', 'L_', 'k0', 'k1', 'b0',
                  'w_lo_beta', 'w_lo_k', 'phi']

# 2d2k types    
est_params_2d2k = models_params['est_params_2d2k']
het_2d2k = copy.deepcopy(pd_rep)
het_2d2k.update({'beta_hyp':1.0, 'L_':est_params_2d2k['L_'],
                 'constrained':True, 'phi':est_params_2d2k['phi']})

    
het_weights_2d2k =  {'w_lo_delta':est_params_2d2k['w_lo_delta'],
                     'w_lo_k':est_params_2d2k['w_lo_k']} 
het_vals_2d2k=  {'d0':est_params_2d2k['d0'], 'd1':est_params_2d2k['d1'],
                  'k0':est_params_2d2k['k0'], 'k1':est_params_2d2k['k1']}
free_args_2d2k = ['L_', 'k0', 'k1', 'd0', 'd1',
                  'w_lo_delta', 'w_lo_k', 'phi']


############################################################
#Apply functions to calculate SEs and write to an out file
############################################################

models = [{'type':'1b1k', 'pd':pd_rep, 'w':None, 'vals':None, 'f_args':free_args_rep},
          {'type':'1b2k', 'pd':het_1b2k, 'w':het_weights_1b2k, 'vals':het_vals_1b2k,
           'f_args':free_args_1b2k},
          {'type':'2b2k', 'pd':het_2b2k, 'w':het_weights_2b2k, 'vals':het_vals_2b2k,
           'f_args':free_args_2b2k},
          {'type':'2b2k_fix_xi', 'pd':het_2b2k_fix_xi, 'w':het_weights_2b2k_fix_xi,
           'vals':het_vals_2b2k_fix_xi, 'f_args':free_args_2b2k_fix_xi},
          {'type':'2b2k_fix_b1', 'pd':het_2b2k_fix_b1, 'w':het_weights_2b2k_fix_b1,
           'vals':het_vals_2b2k_fix_b1, 'f_args':free_args_2b2k_fix_b1},
          {'type':'2d2k', 'pd':het_2d2k, 'w':het_weights_2d2k, 'vals':het_vals_2d2k,
           'f_args':free_args_2d2k}]

def std_errs_wrap(model_dict):
    """Wrapper function to calculate SEs for each model"""
    temp = calc_param_se(model_type=model_dict['type'],
                         p_d = model_dict['pd'],
                         free_args = model_dict['f_args'],
                         w_mat = w_mat, lambda_hat = lambda_hat,
                         emp_hist=  model_dict['pd']['e'], periods=16,
                         cons_start = param.c_plt_start_index ,
                         search_start = param.s_plt_start_index ,
                         norm_time = param.plt_norm_index,
                         het_vals = model_dict['vals'],
                         het_weights = model_dict['w'])
    
    param_index =  temp[2]
    param_index_inv = {v: k for k, v in param_index.iteritems()}
    
    std_errs_mat = np.sqrt(temp[3])
    std_errs = {}
    for key, val in param_index_inv.iteritems():
        std_errs.update({key:std_errs_mat[key][val]})
    return(std_errs)

out = {}
for model in models:
    out.update({model['type']: std_errs_wrap(model)})
df= pd.DataFrame.from_dict(out)  
df.to_excel('../out/SEs.xlsx')

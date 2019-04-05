# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:52:05 2017

@author: Xian_Work
"""


param_path="../Parameters/params_ui.json" 
execfile("prelim.py")
from model_plotting import norm , compute_dist, gen_model_target, sim_plot,  make_base_plot,  mk_mix_agent, gen_evolve_share_series
#For writing output
import xlsxwriter
import pathlib2 as pathlib
import plotnine as p9

#Make path for output if they don't already exist
dirnames = ['../out/1b1k', '../out/OOS', '../out/het_delta',
            '../out/1b2k/GOF_plots', '../out/2b2k/GOF_plots']
for dirname in dirnames:
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)

####################################
#Model Target and Base Plots
####################################
#Data Series to plot against
data_tminus5 = norm(param.JPMC_cons_moments, param.plt_norm_index)
data_tminus5_search = param.JPMC_search_moments


###Use vcv weights###
cons_se_vcv_tminus5 =param.JPMC_cons_SE
search_se_vcv = param.JPMC_search_SE


### Make base plot with data series ###
base_tminus5 = make_base_plot(0, param.c_moments_len,
                              param.moments_len_diff, param.s_moments_len,
                              data_tminus5, data_tminus5_search, 
                              cons_se_vcv_tminus5, search_se_vcv)

##Default param dicts##
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
het_base = copy.deepcopy(pd_base)

### Estimated parameters for main and secondary models
models_params ={}
for path in ["../Parameters/model_params_main.json",
          "../Parameters/model_params_sec.json",
          "../Parameters/model_params_robust_gamma.json"]:
    f= open(path)
    models_params.update(json.load(f))

#########################
#Plotting helper functions
############################
rep_agents_log = []
het_agents_log = []
text_stats = []

def plot_rep(param_dict, label,
             cons_filename, cons_plt_title,
             search_filename, search_plt_title,
             cons_ylim=(0.65, 1.03), search_ylim=(-0.02, 0.35),
             cons_legend_loc=(0.27,0.22), search_legend_loc = (0.27,0.22),
             GOF=False, show_data_CI=False,
             save_stats=False, stats_name=None):
    """Helper function for plotting our basic plot"""
    
    opt_plots=copy.deepcopy(base_tminus5)
    opt_plots.add_agent(label,
                        pd_base['e'],
                        param.c_plt_start_index, 
                        param.s_plt_start_index,
                        param.plt_norm_index,
                        *[(1,param_dict)], verbose=True)
    opt_plots.plot(cons_filename, cons_plt_title,
                   search_filename, search_plt_title,
                   cons_legend_loc = cons_legend_loc,
                   search_legend_loc = search_legend_loc,
                   cons_t0=-5, tminus5=True, show_data_CI = show_data_CI,
                   cons_ylim=cons_ylim, search_ylim=search_ylim,
                   GOF=GOF,)
    
    #save results and parameters
    if save_stats==True: 
        rep_out=copy.deepcopy(opt_plots.agents[0])
        rep_pd=param_dict
        out_rep={'name':stats_name, 'cons_GOF':rep_out['cons_dist'],
                 'search_GOF':rep_out['search_dist'],
                 'k':rep_pd['k'], 'xi':rep_pd['phi'],
                  'delta':rep_pd['beta_var'], 'beta':rep_pd['beta_hyp'],
                 'L_':rep_pd['L_'], 'gamma':rep_pd['rho']}
        rep_agents_log.append(out_rep)
   
def het_agent_plots(het_agent, main_label, type_labels,
                    file_prefix, out_subdir='/',
                    cons_plt_title = None, search_plt_title = None,
                    cons_comp_title = None, search_comp_title = None,
                    shares_title = None, show_data_CI=False,
                    cons_ylim = (0.65, 1.03), search_ylim=(-0.02, 0.35),
                    comps_cons_ylim= (0.47, 1.05), comps_search_ylim = (0, 0.8),
                    cons_legend_loc=(0.27,0.22), search_legend_loc=(0.27,0.22),
                    cons_legend_loc_comps =(0.27,0.22),
                    search_legend_loc_comps =(0.27,0.8),
                    GOF=False,
                    save_stats=False, stats_name=None,
                    save_plots= True):
    '''
    Creates average consumption/search, consumption/search by type, and shares
    plots for a heterogeneous agent. type_labels is a dictionary of index:'label'
    pairs to label each type in order
    
    See model_plotting.py, documentation for the 'plot' function for documentation
    on different options.
    
    '''
    #Filenames and plot titles
    filename_cons = out_subdir + file_prefix + '_cons'
    filename_search = out_subdir + file_prefix + '_search'
    filename_cons_comps = out_subdir + file_prefix + '_cons_components'
    filename_search_comps = out_subdir + file_prefix + '_search_components'
    filename_shares = out_subdir + file_prefix + '_shares'
    
    if save_plots==False:
        filename_cons = None
        filename_search = None
        filename_cons_comps = None
        filename_search_comps = None
        filename_shares = None
    
    if cons_plt_title == None:
        cons_plt_title = 'Spending in Data and Model' + file_prefix
    if search_plt_title == None:
        search_plt_title = 'Job Search in Data and Model' + file_prefix
    if cons_comp_title == None:
        cons_comp_title = 'Spending by type in Model' + file_prefix
    if search_comp_title == None:
        search_comp_title = 'Job Search by type in Model' + file_prefix
    if shares_title == None:
        shares_title = 'Shares in Model'  + file_prefix
        
    

    #Main consumption and search plot
    opt_plots =copy.deepcopy(base_tminus5)
    opt_plots.add_agent(main_label, het_base['e'],
                        param.c_plt_start_index,
                        param.s_plt_start_index,
                        param.plt_norm_index,
                        *het_agent, verbose=True)
    opt_plots.plot(filename_cons, cons_plt_title,
                   filename_search  , search_plt_title,
                   cons_legend_loc = cons_legend_loc,
                   search_legend_loc=search_legend_loc,
                   cons_t0=-5, tminus5=True, show_data_CI = show_data_CI,
                   cons_ylim=cons_ylim, search_ylim=search_ylim,
                   GOF=GOF)
    
    #Components
    comp_dict = {}
    for i in range(len(het_agent)):
        comp_dict.update({type_labels[i]:het_agent[i][1]})
    
    comp_plot=copy.deepcopy(base_tminus5)
    for key,value in comp_dict.iteritems():
        comp_plot.add_agent(key, het_base['e'],
                            param.c_plt_start_index,
                            param.s_plt_start_index,
                            param.plt_norm_index,
                            *[(1,value)])
    comp_plot.plot(filename_cons_comps, cons_comp_title,
                   filename_search_comps , search_comp_title,
                   cons_ylim=comps_cons_ylim, search_ylim=comps_search_ylim,
                   cons_legend_loc=cons_legend_loc_comps, 
                   search_legend_loc=search_legend_loc_comps,
                   cons_t0=-5, GOF=False, tminus5=True, show_data=False) 
    #Shares plot
    fit_out = opt_plots.agents[0]
    shares_plot=copy.deepcopy(base_tminus5)
    shares_dict ={}
    for i in range(len(het_agent)):
        shares_dict.update({i:type_labels[i]})
    shares=fit_out['share_ind']
    for i in range(len(het_agent)):
        share=shares[i]
        shares_plot.add_series(shares_dict[i], share, fit_out['search'])
    shares_plot.plot(filename_shares, shares_title,
                   None , "Job Search",
                   cons_legend_loc = search_legend_loc_comps,
                   search_legend_loc = search_legend_loc_comps,
                   cons_t0=-5,
                   cons_ylim=(0.00,1.06), show_data=False, GOF=False, cons_ylab=' ')
    
    #save results and parameters
    if save_stats==True: 
        for i in range(len(het_agent)):
            type_out = {'name':stats_name, 'type':type_labels[i],
                        'init_share':het_agent[i][0]}
            type_out.update({'cons_GOF':fit_out['cons_dist'],
                             'search_GOF':fit_out['search_dist'],})
            type_pd = het_agent[i][1]
            type_out.update({'k':type_pd['k'], 'xi':type_pd['phi'],
                             'delta':type_pd['beta_var'], 'beta':type_pd['beta_hyp'],
                             'L_':type_pd['L_'], 'gamma':type_pd['rho']})
            het_agents_log.append(type_out)
    
    return fit_out


################################################################################
#################### Models in Table 4 #########################################
################################################################################

##################################
###Representative agent
##################################
pd_rep=copy.deepcopy(pd_base)
est_params_1b1k = models_params['est_params_1b1k']
pd_rep.update(est_params_1b1k)

plot_rep(pd_rep, 'Model: Representative Agent',
         '../out/1b1k/rep_cons', 'Spending in Data and Representative Agent Model',
         '../out/1b1k/rep_search', 'Job Search in Data and Representative Agent Model')
plot_rep(pd_rep, 'Model: Representative Agent',
         '../out/1b1k/rep_cons_GOF', 'Spending in Data and Representative Agent Model',
         '../out/1b1k/rep_search_GOF', 'Job Search in Data and Representative Agent Model',
         GOF=True, save_stats=True,
         cons_legend_loc = (0.33, 0.22), search_legend_loc = (0.33, 0.22))

for name in ['1', '4', '10']:
    pd_temp = copy.deepcopy(pd_rep)
    pd_temp.update(models_params['est_params_1b1k_fix_gamma_' + name])
    plot_rep(pd_temp, 'Model: Representative Agent' + ', gamma = ' + name,
         '../out/1b1k/rep_robust_gamma_' + name, 'Spending in Data and Representative Agent Model',
         '../out/1b1k/rep_robust_gamma_' + name + '_search', 'Job Search in Data and Representative Agent Model',
         GOF=True, save_stats=True, stats_name = '1b1k, gamma =' +name,
         cons_legend_loc = (0.38, 0.22), search_legend_loc = (0.38, 0.22))

#############################
#### 2 types of k  only #####
#############################
#From optimizer
est_params_1b2k = models_params['est_params_1b2k']

#Set up agent
het_1b2k = copy.deepcopy(pd_rep)
het_1b2k.update({'beta_var':est_params_1b2k['beta_var'],'beta_hyp':est_params_1b2k['beta_hyp'],
                 'L_':est_params_1b2k['L_'], 'constrained':True, 'phi':est_params_1b2k['phi'] })

#weights 
weights_1b2k = (est_params_1b2k['w_lo_k'], 1- est_params_1b2k['w_lo_k'])
params_1b2k = ('k', )
vals_1b2k = ((est_params_1b2k['k0'], ),
             (est_params_1b2k['k1'], ),)

het_1b2k_agent = mk_mix_agent(het_1b2k, params_1b2k, vals_1b2k, weights_1b2k)
het_agent_labels = {0:'Low Search Cost', 1:'High Search Cost'}
het_agent_plots(het_1b2k_agent, 'Model: Standard',
                het_agent_labels, '1b2k', out_subdir='1b2k/',
                cons_plt_title = "Spending in Data and Standard Model",
                search_plt_title = "Job Search in Data and Standard Model",
                cons_comp_title = "Spending by type, Standard Model",
                search_comp_title = "Job Search by type, Standard Model",
                shares_title = "Shares, Standard Model", show_data_CI = True,
                cons_legend_loc = (0.23, 0.22), search_legend_loc = (0.23, 0.22))
het_agent_plots(het_1b2k_agent, 'Model: Baseline',
                het_agent_labels, '1b2k', out_subdir='1b2k/GOF_plots/',
                cons_plt_title = "Spending in Data and Standard Model",
                search_plt_title = "Job Search in Data and Standard Model",
                cons_comp_title = "Spending by type, Standard Model",
                search_comp_title = "Job Search by type, Standard Model",
                shares_title = "Shares, Standard Model",                                                
                GOF=True, save_stats=True, show_data_CI = True,
                stats_name = '2_k_types' )

#############################################
#### 2 types for beta and 2 types for k #####
#############################################
robustness_2b2k=[('est_params_2b2k', '2b2k', "Heterogeneous Beta"),
                 ('est_params_2b2k_fix_xi', '2b2k_fix_xi', "Heterogeneous Beta, xi=1.0"),
                 ('est_params_2b2k_fix_b1', '2b2k_fix_b1', "Heterogeneous Beta, B_hi=1.0"),]

#To plot both models together
plot_2b2k_both = copy.deepcopy(base_tminus5)

#Plot each model separately
for model in robustness_2b2k:
    est_params_2b2k = models_params[model[0]]
    if model[0] == 'est_params_2b2k':   
        show_data_CI_2b2k = True
    else:
        show_data_CI_2b2k = False
    
    het_2b2k = copy.deepcopy(pd_rep)
    het_2b2k.update({'beta_var':est_params_2b2k['beta_var'], 'L_':est_params_2b2k['L_'],
                     'constrained':True, 'phi':est_params_2b2k['phi']})
    
    k0 = est_params_2b2k['k0']
    k1 = est_params_2b2k['k1']
    
    b0 = est_params_2b2k['b0']
    b1 = est_params_2b2k['b1']
    
    
    #weights 
    w_lo_k = est_params_2b2k['w_lo_k']
    w_hi_k = 1 - w_lo_k
    w_lo_beta = est_params_2b2k['w_lo_beta']
    w_hi_beta = 1 - w_lo_beta
    
    w_b0_k0 = w_lo_k * w_lo_beta
    w_b1_k0 = w_lo_k * w_hi_beta
    w_b0_k1 = w_hi_k * w_lo_beta
    w_b1_k1 = w_hi_k * w_hi_beta
    
    #weights 
    weights_2b2k = (w_b0_k0, w_b1_k0, w_b0_k1, w_b1_k1)
    params_2b2k =  ('beta_hyp', 'k' )
    vals_2b2k = ((b0, k0),
                 (b1, k0 ),
                 (b0, k1),
                 (b1, k1 ))
    
    het_2b2k_agent = mk_mix_agent(het_2b2k, params_2b2k, vals_2b2k, weights_2b2k)

    het_agent_labels = {0:'Hyperbolic, Low Search Cost', 1:'Exponential, Low Search Cost',
                        2:'Hyperbolic, High Search Cost',3:'Exponential, High Search Cost'}
    het_agent_plots(het_2b2k_agent, 'Model: Heterogeneity in Beta',
                    het_agent_labels, model[1], out_subdir='2b2k/',
                    cons_plt_title = "Spending in Data and Heterogeneous Beta Model",
                    search_plt_title = "Job Search in Data and Heterogeneous Beta Model",
                    cons_comp_title = "Spending by type, Heterogeneous Beta Model",
                    search_comp_title = "Job Search by type, Heterogeneous Beta Model",
                    shares_title = "Shares, Heterogeneous Beta Model",
                    show_data_CI = show_data_CI_2b2k,
                    cons_legend_loc = (0.29, 0.22), search_legend_loc = (0.29, 0.22),
                    cons_legend_loc_comps = (0.29, 0.25), search_legend_loc_comps = (0.29, 0.7), )
    het_agent_plots(het_2b2k_agent, 'Model: Heterogeneity in Beta',
                    het_agent_labels, model[1], out_subdir='2b2k/GOF_plots/',
                    cons_plt_title = "Spending in Data and Heterogeneous Beta Model",
                    search_plt_title = "Job Search in Data and Heterogeneous Beta Model",
                    cons_comp_title = "Spending by type, Heterogeneous Beta Model",
                    search_comp_title = "Job Search by type, Heterogeneous Beta Model",
                    shares_title = "Shares, Heterogeneous Beta Model",
                    GOF=True, save_stats=True, stats_name = model[1],
                    show_data_CI = show_data_CI_2b2k,
                    cons_legend_loc_comps = (0.29, 0.25), search_legend_loc_comps = (0.29, 0.7),
                    cons_legend_loc = (0.32, 0.22), search_legend_loc = (0.32, 0.22))
        
    #Combined plot with 2b2k and 2b2k, fixed xi=1.0 models
    if model[1] == '2b2k' or model[1] == '2b2k_fix_xi':
        plot_2b2k_both.add_agent('Model: ' + model[2], het_base['e'],
                                 param.c_plt_start_index,
                                 param.s_plt_start_index,
                                 param.plt_norm_index,
                                 *het_2b2k_agent)
        
    ###Stat for text
    if model[1] == '2b2k'  :
        agent_series = gen_evolve_share_series(het_base['e'],
                                                param.c_plt_start_index,
                                                param.s_plt_start_index,
                                                base_tminus5.periods,
                                                param.plt_norm_index,
                                                *het_2b2k_agent, verbose=True)
        shares_b0k0 = list(agent_series['share_ind'][0])
        shares_b0k1 = list(agent_series['share_ind'][2])
        m5_share_myopic = shares_b0k0[10] + shares_b0k1[10]
        text_stats.append(('''By month 5 - the last month of UI benefits - the
                           myopic types are XXX percent of the population''',
                               np.round(100*m5_share_myopic,decimals=0)))

plot_2b2k_both.plot('/2b2k/2b2k_with_fixed_xi_cons', "Spending in Data and Heterogeneous Beta Models",
                    '/2b2k/2b2k_with_fixed_xi_search', "Job Search in Data and Heterogeneous Beta Models",
                    cons_legend_loc =(0.34,0.22), search_legend_loc =(0.34,0.22),
                    cons_ylim=(0.65,1.03), GOF=True)        
#######################################
#############Florida OOS############
#######################################
###generate the standard agents 
FL_2k=copy.deepcopy(het_1b2k)
FL_2k['z_vals']=np.array(param.z_vals_FL)
FL_2k_agent = mk_mix_agent(FL_2k, params_1b2k, vals_1b2k, weights_1b2k)

###generate the spender-saver agents 
est_params_2b2k = models_params['est_params_2b2k']
FL_2b2k = copy.deepcopy(pd_rep)
FL_2b2k['z_vals']=np.array(param.z_vals_FL)
FL_2b2k.update({'beta_var':est_params_2b2k['beta_var'], 'L_':est_params_2b2k['L_'],
                 'constrained':True, 'phi':est_params_2b2k['phi']})
het_2b2k = copy.deepcopy(pd_rep)
het_2b2k.update({'beta_var':est_params_2b2k['beta_var'], 'L_':est_params_2b2k['L_'],
                 'constrained':True, 'phi':est_params_2b2k['phi']})

k0 = est_params_2b2k['k0']
k1 = est_params_2b2k['k1']
b0 = est_params_2b2k['b0']
b1 = est_params_2b2k['b1']

#weights 
w_lo_k = est_params_2b2k['w_lo_k']
w_hi_k = 1 - w_lo_k
w_lo_beta = est_params_2b2k['w_lo_beta']
w_hi_beta = 1 - w_lo_beta

w_b0_k0 = w_lo_k * w_lo_beta
w_b1_k0 = w_lo_k * w_hi_beta
w_b0_k1 = w_hi_k * w_lo_beta
w_b1_k1 = w_hi_k * w_hi_beta

weights_2b2k = (w_b0_k0, w_b1_k0, w_b0_k1, w_b1_k1)
params_2b2k =  ('beta_hyp', 'k' )
vals_2b2k = ((b0, k0),
             (b1, k0 ),
             (b0, k1),
             (b1, k1 ))

#Generate the 6-month and the Flordia spender-saver (2b2k) agent 
het_2b2k_agent =  mk_mix_agent(het_2b2k, params_2b2k, vals_2b2k, weights_2b2k)
FL_2b2k_agent = mk_mix_agent(FL_2b2k, params_2b2k, vals_2b2k, weights_2b2k)

####################
#FL and NJ Targets 
######################
FL_cons_data = param.JPMC_cons_moments_FL
FL_cons_data=norm(FL_cons_data, param.plt_norm_index)
FL_cons_se=param.JPMC_cons_SE_FL

FL_search_data = param.JPMC_search_moments_FL
FL_search_se= [1] * param.s_moments_len

FL_target_plot=make_base_plot(0, param.c_moments_len,
                              param.moments_len_diff,param.s_moments_len,
                              FL_cons_data, FL_search_data,
                              FL_cons_se, FL_search_se)

#####################
#FL plots
#####################    
opt_plots=copy.deepcopy(FL_target_plot)
name="Model: Standard"
opt_plots.add_agent(name,FL_2k['e'],
                    param.c_plt_start_index,
                    param.s_plt_start_index,
                    param.plt_norm_index,
                    *FL_2k_agent, verbose=True)
name="Model: Heterogeneity in Beta"
opt_plots.add_agent(name,FL_2b2k['e'],
                    param.c_plt_start_index,
                    param.s_plt_start_index,
                    param.plt_norm_index,
                    *FL_2b2k_agent)
opt_plots.plot("/OOS/FL_cons", "Spending in Data and in Models,\nOut of Sample Test With Low-Benefit State Florida" ,
               "/OOS/FL_search"  ,  "Job Search in Data and in Models,\nOut of Sample Test With Low-Benefit State Florida",
                cons_ylim = (0.65, 1.03), search_ylim = (0, 0.55),
                search_legend_loc = (0.27,0.8), GOF=True,
                cons_t0=-5, show_data_CI = True, florida=True)
FL_2k_cons = opt_plots.agents[0]['cons']
FL_2b2k_cons = opt_plots.agents[1]['cons']


###############################################################################
#############################  Appendix Plots #################################
###############################################################################

#####################################
### Standard Model Corner Cases
#####################################

##Standard Model (2k) with no initial assets
pd_2k_no_assets = copy.deepcopy(het_1b2k)
pd_2k_no_assets.update({'L_':0.0})
pd_2k_no_assets['a0']=0.0
pd_2k_no_assets['e']=[0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8] #this emp history starts from t=-3
pd_2k_no_assets_agent = mk_mix_agent(pd_2k_no_assets, params_1b2k, vals_1b2k, weights_1b2k)

base_tminus2 = make_base_plot(0, param.c_moments_len - 3,
                              param.moments_len_diff -3 , param.s_moments_len,
                              data_tminus5[3:], data_tminus5_search[3:], 
                              cons_se_vcv_tminus5[3:], search_se_vcv)
base_tminus2.add_agent('Model: Standard, Initial Assets = 0',
                       pd_2k_no_assets['e'], 0,2,0, *pd_2k_no_assets_agent)
base_tminus2.plot('1b2k/1b2k_no_init_assets_cons', "Spending in Data and Standard Model", 
                  '1b2k/1b2k_no_init_assets_search', "Job Search in Data and Standard Model",
                  cons_t0=-2, cons_ylim=(0.55,1.03),)


##Standard Model (2k) with impatience -  hyperbolic
params_1b2k_hyp_impatient = models_params['robust_hyp_impatient']
pd_1b2k_hyp_impatient = copy.deepcopy(het_1b2k)
pd_1b2k_hyp_impatient.update({'beta_hyp':params_1b2k_hyp_impatient['beta_hyp']})
hyp_impatient_agent = mk_mix_agent(pd_1b2k_hyp_impatient, params_1b2k,
                                           vals_1b2k, weights_1b2k)

imp_beta = str(round(pd_1b2k_hyp_impatient['beta_hyp'],1))
het_agent_labels = {0:'Low Search Cost', 1:'High Search Cost'}
het_agent_plots(hyp_impatient_agent, 'Model: Beta = ' + imp_beta,
                het_agent_labels, '1b2k_hyp_impatient', out_subdir='1b2k/',
                cons_plt_title = "Spending in Data and Model with Quasi-Hyperbolic",
                search_plt_title = "Job Search in Data and Model with Quasi-Hyperbolic",
                cons_comp_title = "Spending by type, Model with Quasi-Hyperbolic",
                search_comp_title = "Job Search by type, Model with Quasi-Hyperbolic",
                shares_title = "Shares, Model with Quasi-Hyperbolic",
                cons_legend_loc = (0.23, 0.22), search_legend_loc = (0.23, 0.22),
                cons_ylim=(0.5, 1.03))

##Standard Model (2k) with impatience -  exponential
opt_plots=copy.deepcopy(base_tminus5)
for std_exp_imp in [(0,'robust_exp_impatient_0'), (1,'robust_exp_impatient_1')]:
    params_1b2k_exp_impatient = models_params[std_exp_imp[1]]
    pd_1b2k_exp_impatient = copy.deepcopy(het_1b2k)
    pd_1b2k_exp_impatient.update({'beta_var':params_1b2k_exp_impatient['beta_var']})
    exp_impatient_agent = mk_mix_agent(pd_1b2k_exp_impatient, params_1b2k,
                                               vals_1b2k, weights_1b2k)
    
    opt_plots.add_agent("Model: Delta = {0:.1f}".format(params_1b2k_exp_impatient['beta_var']),
                        het_base['e'],
                        param.c_plt_start_index,
                        param.s_plt_start_index,
                        param.plt_norm_index,
                        *exp_impatient_agent, verbose=True)
    
opt_plots.plot('1b2k/1b2k_exp_impatient_cons', 'Spending in Data and Model with Impatient Exponential',
               None, 'Job Search with Permanent Income Loss',
               cons_t0=-5, tminus5=True, cons_ylim=(0.53, 1.03), cons_legend_loc = (0.23, 0.22),)

##############################################################
#### Heterogeneity in Delta 
##############################################################
het_2d2k_est_params = models_params['est_params_2d2k']

#Set up agent
het_2d2k = copy.deepcopy(pd_rep)
het_2d2k.update({'L_':het_2d2k_est_params['L_'],
              'phi':het_2d2k_est_params['phi'], 'beta_hyp':1.0},)

k0 = het_2d2k_est_params['k0']
k1 = het_2d2k_est_params['k1']
d0 = het_2d2k_est_params['d0']
d1 = het_2d2k_est_params['d1']

#weights 
w_lo_k = het_2d2k_est_params['w_lo_k']
w_hi_k = 1 - w_lo_k
w_lo_delta = het_2d2k_est_params['w_lo_delta']
w_hi_delta = 1 - w_lo_delta

w_d0_k0 = w_lo_k * w_lo_delta
w_d1_k0 = w_lo_k * w_hi_delta
w_d0_k1 = w_hi_k * w_lo_delta
w_d1_k1 = w_hi_k * w_hi_delta

#weights 
weights_het_2d2k = (w_d0_k0, w_d1_k0, w_d0_k1, w_d1_k1)
params_het_2d2k = ('beta_var', 'k')
vals_het_2d2k = ((d0, k0),
                 (d1, k0 ),
                 (d0, k1),
                 (d1, k1 ))

het_2d2k_agent = mk_mix_agent(het_2d2k, params_het_2d2k, vals_het_2d2k, weights_het_2d2k)
het_agent_labels = {0:'Low Delta, Low Search Cost', 1:'High Delta, Low Search Cost',
                    2:'Low Delta, High Search Cost', 3:'High Delta, High Search Cost',}
het_agent_plots(het_2d2k_agent, 'Model: Heterogeneity in Exponential Discount Factor',
                het_agent_labels, 'het_2d2k',
                cons_plt_title = "Spending in Data and \nModel with Heterogeneity in Exponential Discount Factor",
                search_plt_title = "Job Search in Data and \nModel with Heterogeneity in Exponential Discount Factor",
                cons_comp_title = "Spending by type,\nModel with Heterogeneity in Exponential Discount Factor",
                search_comp_title = "Job Search by type,\nModel with Heterogeneity in Exponential Discount Factor",
                shares_title = "Shares, Model with Heterogeneity in Exponential Discount Factor",
                GOF=False, out_subdir='het_delta/',
                cons_legend_loc = (0.38, 0.22), search_legend_loc = (0.38, 0.22),
                cons_legend_loc_comps = (0.29, 0.25), search_legend_loc_comps = (0.29, 0.7),)
het_agent_plots(het_2d2k_agent, 'Model: Heterogeneity in Exponential Discount Factor',
                het_agent_labels, 'het_2d2k',
                cons_plt_title = "Spending in Data and \nModel with Heterogeneity in Exponential Discount Factor",
                search_plt_title = "Job Search in Data and \nModel with Heterogeneity in Exponential Discount Factor",
                cons_comp_title = "Spending by type,\nModel with Heterogeneity in Exponential Discount Factor",
                search_comp_title = "Job Search by type,\nModel with Heterogeneity in Exponential Discount Factor",
                shares_title = "Shares, Model with Heterogeneity in Exponential Discount Factor",
                GOF=True, out_subdir='het_delta/GOF_plots/', save_stats=True, stats_name='2d2k_types',
                cons_legend_loc = (0.41, 0.22), search_legend_loc = (0.41, 0.22),
                cons_legend_loc_comps = (0.29, 0.25), search_legend_loc_comps = (0.29, 0.7))

####Compare 1b2k, 2d2k, 2b2k
plots_2b2k_2d2k = copy.deepcopy(base_tminus5)
for agent in [('Heterogeneity in Beta', het_2b2k_agent), ('Heterogeneity in Delta', het_2d2k_agent),
              ('Standard', het_1b2k_agent)]:
    plots_2b2k_2d2k.add_agent('Model: '+ agent[0], het_base['e'],
                              param.c_plt_start_index,
                              param.s_plt_start_index,
                              param.plt_norm_index,
                              *agent[1], verbose=True)
plots_2b2k_2d2k.plot('../out/compare_beta_delta_cons',
                     'Spending in Data and Alternative Models with Heterogeneity',
                     '../out/compare_beta_delta_search',
                     'Job Search in Data and Alternative Models with Heterogeneity',
                     GOF=False, cons_ylim = (0.7, 1.03),
                     cons_legend_loc = (0.29, 0.25), search_legend_loc = (0.29, 0.25))

#Bar plot with GOFs
df = pd.DataFrame({"Model":['Standard','Heterogeneity in Beta', 'Heterogeneity in Delta'],
                   "Consumption GOF":[349, 99, 148]})
fig = pd.melt(df, id_vars=['Model'])
pp = p9.ggplot(fig, p9.aes(x='Model', y='value')) +\
     p9.geom_bar(stat = 'identity', width = 0.5) +\
     p9.labs(title='Consumption Goodness of Fit in Models with Heterogeneity' , y="Consumption GOF") 
     
pp.save(filename="../out/compare_beta_delta_GOFs.pdf", plot=pp, width =7, height = 4, verbose=False)

         
################################################
#######Permanent Income Loss###################
###############################################
perm_inc_loss_params = models_params['perm_inc_loss_params']
####Set up Pi for permanent income loss#####
sep_rate = 1- param.Pi[0][0]
ex_jf_rates = perm_inc_loss_params['ex_jf_rates']

Pi = np.array(np.zeros((17,17)))
Pi[0,0], Pi[0,1] = 1-sep_rate, sep_rate
for i in range(len(ex_jf_rates)):
    if i <7:
        jf_rate = ex_jf_rates[i]
        Pi[i+1,0], Pi[i+1,i+2] = jf_rate, 1-jf_rate 
    if i ==7: #exhaustion and permanent income loss
        Pi[i+1,i+2], Pi[i+1,i+1] = jf_rate, 1-jf_rate

#Employed state after having experienced exhaustion
Pi[9,9], Pi[9,10] = 1-sep_rate, sep_rate

#UI states after having experienced exhaustion
for i in range(0,7):
    jf_rate = ex_jf_rates[i]
    
    #Transition from UI into employment, after having experienced exhaustion once
    if i<6:        
        Pi[i+10, 9], Pi[i+10, i+11] = jf_rate, 1-jf_rate
    #Transition into exhaustion
    if i ==6: 
        Pi[i+10, 9], Pi[i+10, 8] = jf_rate, 1-jf_rate
        
Pi_perm_loss = copy.deepcopy(Pi)

#####Set up Pi for permanent income loss with uncertainity####
jf_rate_exhaust = perm_inc_loss_params['jf_rate_exhaust']
Pi_uncertn_loss = copy.deepcopy(Pi)
Pi_uncertn_loss[8,8], Pi_uncertn_loss[8,0], Pi_uncertn_loss[8,9] = \
1-jf_rate_exhaust, jf_rate_exhaust/2, jf_rate_exhaust/2

####Z vals for permanent income loss####
z_vals_perm_loss = np.array(perm_inc_loss_params['z_vals_perm_loss'])       

###Set up agents with permanent income loss
pd_perm_loss = copy.deepcopy(pd_rep)
pd_perm_loss['solve_V']=False
pd_perm_loss['solve_search']=False
pd_perm_loss['z_vals']= z_vals_perm_loss

pd_perm_loss["Pi_"] = np.zeros((param.TT+1,param.a_size+1,17,17))
for t in range(param.TT+1):
    for a_index in range(param.a_size+1):
        pd_perm_loss['Pi_'][t][a_index] = Pi_perm_loss
        
pd_uncertn_loss = copy.deepcopy(pd_perm_loss)
for t in range(param.TT+1):
    for a_index in range(param.a_size+1):
        pd_uncertn_loss['Pi_'][t][a_index] = Pi_uncertn_loss
        
#Plot permant income loss
opt_plots=copy.deepcopy(base_tminus5)
opt_plots.add_agent("Lose 10% at Exhaust (Certain)", pd_perm_loss['e'],
                    param.c_plt_start_index,
                    param.s_plt_start_index,
                    param.plt_norm_index,
                    *[(1,pd_perm_loss)], verbose=True)
opt_plots.add_agent("Lose 10% at Exhaust (Uncertain)", pd_uncertn_loss['e'],
                    param.c_plt_start_index,
                    param.s_plt_start_index,
                    param.plt_norm_index,
                    *[(1,pd_uncertn_loss)], verbose=True)
opt_plots.plot('perm_inc_loss_cons', 'Spending in Data and Permanent Income Loss Model',
               None, 'Job Search with Permanent Income Loss',
               cons_t0=-5, tminus5=True, cons_ylim=(0.67, 1.03))


################################################
#######Overconfidence#########################
###############################################
est_overopt_params = models_params['est_overopt_params']
### Overconfidence that fits drops at exhaustion ###
Pi_optimistic = copy.deepcopy(param.Pi)

jf_ui = est_overopt_params['jf_ui']
jf_mo_7 = est_overopt_params['jf_mo_7']
jf_exhaust = est_overopt_params['jf_exhaust']

for i in range(1, len(Pi_optimistic)):
    if i<7:
        Pi_optimistic[i,0], Pi_optimistic[i,i+1] = jf_ui, 1-jf_ui
    if i ==7:
        Pi_optimistic[i,0], Pi_optimistic[i,i+1] = jf_mo_7, 1-jf_mo_7
    if i > 7:
        Pi_optimistic[i,0], Pi_optimistic[i,i] = jf_exhaust, 1-jf_exhaust

#Set up agent
pd_optimistic = copy.deepcopy(pd_rep)        
pd_optimistic['solve_V']=False
pd_optimistic['solve_search']=False
for t in range(param.TT+1):
    for a_index in range(param.a_size+1):
        pd_optimistic['Pi_'][t][a_index] = Pi_optimistic
        
#Plot fitted overconfidence
dum_plot = copy.deepcopy(base_tminus5)        
dum_plot.add_agent("Over-optimism Estimated to fit Spending", pd_optimistic['e'],
                   param.c_plt_start_index,
                   param.s_plt_start_index,
                   param.plt_norm_index,
                   *[(1,pd_optimistic)], verbose=True)
cons_optimistic = dum_plot.agents[0]['cons']
search_optimistic =np.concatenate(([jf_ui ]*5, [jf_mo_7], [jf_exhaust]*5))
        
opt_plots=copy.deepcopy(base_tminus5)
opt_plots.add_series("Over-Optimism Estimated to fit Spending",
                    cons_optimistic, search_optimistic)
opt_plots.plot('overconfidence_cons', 'Spending in Data and \nModel with Overconfident Job-finding Beliefs',
               'overconfidence_search', 'Job Search in Data and Model with Overconfident Beliefs',
               cons_t0=-5, tminus5=True,
               cons_ylim =(0.63, 1.03), search_ylim=(0,0.8 ),
               search_legend_loc = (0.33,0.75))



#### Persistent overconfidence in job-finding probablity
persist_overopt_params = models_params['persist_overopt_params']

Pi_opt_persist = copy.deepcopy(param.Pi)
jf_rate_avg = (np.sum(Pi_opt_persist, axis = 0)[0] - Pi_opt_persist[0][0])/Pi_opt_persist.shape[0]
jf_rate_opt_const = jf_rate_avg + persist_overopt_params['overopt_const']
jf_rate_opt_pct = jf_rate_avg * persist_overopt_params['overopt_mult']
jf_rate_opt = jf_rate_opt_const

for i in range(1, Pi_opt_persist.shape[0]):
    Pi_opt_persist[i][0] = jf_rate_opt
    Pi_opt_persist[i][min(i+1,Pi_opt_persist.shape[0]-1)] = 1- jf_rate_opt

#Setup agent
pd_opt_persist = copy.deepcopy(pd_rep)
pd_opt_persist['solve_V'] = False
pd_opt_persist['solve_search'] = False
for t in range(param.TT+1):
    for a_index in range(param.a_size+1):
        pd_opt_persist['Pi_'][t][a_index] = Pi_opt_persist

dum_plot = copy.deepcopy(base_tminus5)
dum_plot.add_agent('Calibrated Over-optimism', pd_opt_persist['e'],
                   param.c_plt_start_index,
                   param.s_plt_start_index,
                   param.plt_norm_index,
                   *[(1,pd_opt_persist)], verbose=True)
cons_optimistic_persist = dum_plot.agents[0]['cons']
search_optimistic_persist =np.array([jf_rate_opt]*11)

#Plot calibrated overconfidence
opt_plots=copy.deepcopy(base_tminus5)
opt_plots.add_series("Model: Calibrated Over-Optimism",
                    cons_optimistic_persist, search_optimistic_persist)
opt_plots.plot('overconfidence_persist_cons',
               'Spending in Data and \nModel with Overconfident Job-finding Beliefs',
               'overconfidence_persist_search',
               'Job Search in Data and Model with Overconfident Beliefs',
               cons_t0=-5, tminus5=True,
               cons_ylim =(0.5, 1.03), search_ylim=(0,0.8 ),
               search_legend_loc = (0.29,0.75))
opt_plots.add_agent('Model: Representative Agent', pd_rep['e'],
                    param.c_plt_start_index,
                    param.s_plt_start_index,
                    param.plt_norm_index,
                    *[(1,pd_rep)], verbose=True)
opt_plots.plot('overconfidence_persist_cons_1',
               'Spending in Data and \nModel with Overconfident Job-finding Beliefs',
               'overconfidence_persist_search_1',
               'Job Search in Data and Model with Overconfident Beliefs',
               cons_t0=-5, tminus5=True,
               cons_ylim =(0.5, 1.03), search_ylim=(0,0.8 ),
               search_legend_loc = (0.29,0.75))


###############################
### Different values of gamma ###
###############################
#### Plot not created or used in paper ###
robust_gamma_1 = models_params['est_params_1b2k_fix_gamma_1']
robust_gamma_4 = models_params['est_params_1b2k_fix_gamma_4']
robust_gamma_10 = models_params['est_params_1b2k_fix_gamma_10']
robust_gamma_models = [robust_gamma_1, robust_gamma_4, robust_gamma_10]

opt_plots=copy.deepcopy(base_tminus5)
legend_title = 'Model: Baseline, Gamma ={0:.3f}, Delta = {1:.3f}'.format(het_1b2k['rho'], het_1b2k['beta_var'], )
opt_plots.add_agent(legend_title, pd_base['e'],
                    param.c_plt_start_index,
                    param.s_plt_start_index,
                    param.plt_norm_index,
                    *het_1b2k_agent, verbose=True)


for model in robust_gamma_models:
    p_d = copy.deepcopy(het_1b2k)
    p_d.update({'beta_var':model['beta_var'], 'rho':model['rho']})

    agent = mk_mix_agent(p_d, params_1b2k, vals_1b2k, weights_1b2k)
    legend_title = 'Model: Gamma ={0:.3f}, Delta = {1:.3f}'.format(p_d['rho'], p_d['beta_var'], )

    opt_plots.add_agent(legend_title, pd_base['e'],
                        param.c_plt_start_index,
                        param.s_plt_start_index,
                        param.plt_norm_index,
                        *agent, verbose=True)

#### Log results ###
df = pd.DataFrame(het_agents_log)
df = df[['name', 'cons_GOF', 'search_GOF', 'type', 'init_share',
        'beta', 'delta', 'L_', 'k', 'xi']]
df.to_excel('../out/het_agents_log.xlsx')

df = pd.DataFrame(rep_agents_log)
df = df[['name', 'cons_GOF', 'search_GOF', 
        'beta', 'delta', 'L_', 'k', 'xi']]
df.to_excel('../out/rep_agents_log.xlsx')

df = pd.DataFrame(text_stats)
df.to_excel('../out/stats_for_text.xlsx')


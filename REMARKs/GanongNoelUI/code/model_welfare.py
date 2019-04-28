# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:16:24 2017

@author: Xian_Work
"""


param_path="../Parameters/params_ui.json" 
execfile("prelim.py")
from agent_history import rand_hist, agent_history_endo_js
from solve_search import search, search_cost
from model_plotting import opt_dict_csv_in, mk_mix_agent, norm
import matplotlib.pyplot as plt


#Initialize parameters
#########################3 initialize parameters
###This is from our data moments; post-exhaustion is an average
Pi_extend = param.Pi_extend

### Estimated parameters for other models
models_params ={}
for path in ["../Parameters/model_params_main.json",
          "../Parameters/model_params_sec.json",
          "../Parameters/model_params_robust_gamma.json"]:
    f= open(path)
    models_params.update(json.load(f))

z_exhaust = param.z_vals[-1]
#####################Set up rep buffer-stock  agent########################
pd_base = {"a0": param.a0_data, "T_series":T_series, "T_solve":param.TT, 
               "e":param.e_extend, 
               "beta_var":param.beta, "beta_hyp": param.beta_hyp, "a_size": param.a_size,
               "rho":param.rho, "verbose":False, "L_":param.L, 
               "constrained":param.constrained, "Pi_":np.zeros((param.TT+1,param.a_size+1,9,9)),
               "z_vals" : param.z_vals, "R" : param.R, "Rbor" : param.R, 
               "phi": param.phi, "k":param.k, "spline_k":param.spline_k, "solve_V": True,
               "solve_search": True}
pd_base['Pi_'] = np.zeros((param.TT+1,param.a_size+1,10,10))
for t in range(param.TT+1):
    for a_index in range(param.a_size+1):
        pd_base['Pi_'][t][a_index] = Pi_extend
pd_base['T_series']=len(pd_base['e'])-1
het_base = copy.deepcopy(pd_base)

est_params_1b1k = models_params['est_params_1b1k']
pd_rep = copy.deepcopy(pd_base)
pd_rep.update(est_params_1b1k)

#### Rep agent with different gamma
est_gamma_1 = models_params['est_params_1b1k_fix_gamma_1']
pd_gamma_1 = copy.deepcopy(pd_rep)
pd_gamma_1.update({'rho':est_gamma_1['rho'], 'beta_var':est_gamma_1['beta_var']})

est_gamma_4 = models_params['est_params_1b1k_fix_gamma_4']    
pd_gamma_4 = copy.deepcopy(pd_rep)
pd_gamma_4.update({'rho':est_gamma_4['rho'], 'beta_var':est_gamma_4['beta_var'],})


####################################
### Standard Model #########
####################################
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

####################################
### Heterogeneity 2beta x 2k########
####################################
agents_2b2k = ['est_params_2b2k',
               "est_params_2b2k_fix_xi",]
agents_2b2k_dict = {}


for est_type in agents_2b2k:
    est_params = models_params[est_type]
    
    het_2b2k = copy.deepcopy(pd_rep)
    het_2b2k.update({'beta_var':est_params['beta_var'], 'L_':est_params['L_'],
                     'constrained':True, 'phi':est_params['phi']})
    
    k0 = est_params['k0']
    k1 = est_params['k1']
    
    b0 = est_params['b0']
    b1 = est_params['b1']
    
    
    #weights 
    w_lo_k = est_params['w_lo_k']
    w_hi_k = 1 - w_lo_k
    w_lo_beta = est_params['w_lo_beta']
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
    
    agent = mk_mix_agent(het_2b2k, params_2b2k, vals_2b2k, weights_2b2k)
    agents_2b2k_dict.update({est_type:agent})

het_2b2k_agent = agents_2b2k_dict['est_params_2b2k']
het_2b2k_agent_fix_xi = agents_2b2k_dict['est_params_2b2k_fix_xi']

##############################################################################
###### Set up random array for simulations and base benefit schedule #########
##############################################################################
rand_array=rand_hist(240,1500)
z_vals_extend = np.insert(param.z_vals, 8, param.z_vals[-1])

#set up money-metric increase for comparison
z_mm = copy.deepcopy(z_vals_extend)
z_mm = z_mm + np.repeat(0.01, len(z_vals_extend))

out_list=[]

ui_cons_data = param.JPMC_cons_moments_ui
ui_cons_data_norm = norm(ui_cons_data, 0)

##################### Helper functions ######################
def run_sim(p_d, z_vals, ue_inc, ex_search):
    """Simulates histories"""
    agent = agent_history_endo_js(p_d, z_vals,
                                  rand_array, ex_search=ex_search) 
    agent.sim_hist()
    agent.calc_welfare()
    agent.calc_ex(ue_inc)
    return agent

def run_ss(SS_agent, z_vals, ue_inc, ex_search):
    """Simulates histories for an SS agent"""
    comp_sims=[]
    
    for agent in SS_agent:
        weight = agent[0]
        p_d = agent[1]
        sim = run_sim(p_d, z_vals, ue_inc, ex_search)
        comp_sims.append((weight,sim))
    return comp_sims

def eval_ss(ss_mod, ss_base, ss_mm):
    """Evaluates stats for the SS agent, type by type
    
    Args:
        ss_mod: list of (weight, simulated history) tuples
        ss_base: list of (weight, simulated history) tuples, base case
        ss_mm: list of (weight, simulated history) tuples, money-metric increase
    
    """
    if not (len(ss_mod) == len(ss_base) == len(ss_mm)):
        ValueError('Unequal number of agents!')
    
    parts_dict={}
    comb_dict={'dW_smooth': 0,
               'dW_search': 0,
               'dW_tot': 0,
               'mod_ui_ex': 0,
               'mod_tax_rev': 0,
               'base_ui_ex': 0,
               'base_tax_rev': 0,}
    
    
    dW_smooth_num = 0.0
    dW_search_num = 0.0
    
    dW_denom = 0.0
    
    for i in range(len(ss_mod)):
        agent_stats={'type': i}
        
        if not (ss_mod[i][0] == ss_base[i][0]  == ss_mm[i][0]):
            ValueError('Weights not equal for {0:.f}-th agent'.format(i))
        weight = ss_mod[i][0]
        mod = ss_mod[i][1]
        base = ss_base[i][1]
        mm = ss_mm[i][1]
        
        #############Diagnostic - smoothing gains and losses decomposed ######
        c_mod = mod.c_hist
        e_mod = mod.e_hist
        emp_mask = np.where(e_mod==0)
        c_mod_notax = copy.deepcopy(c_mod)
        c_mod_notax[emp_mask] = 1.0
        
        
        ######################################################################
        
        #compute welfare change for modification and for money-metric change
        dW_smooth = mod.u_cons - base.u_cons
        dW_search = mod.u_search - base.u_search
        dW_mod = mod.u_tot - base.u_tot
        
        #Money-metric change; denominator
        dW_mm = mm.u_tot - base.u_tot
        
        #Save stats for each agent
        agent_stats.update({'dW_smooth': dW_smooth / dW_mm,
                            'dW_search': dW_search / dW_mm,
                            'dW_tot' : dW_mod / dW_mm})
        #Taxes and revenue
        agent_stats.update({'mod_ui_ex': mod.ui_ex,
                            'mod_tax_rev': mod.tax_rev,
                            'base_ui_ex':base.ui_ex,
                            'base_tax_rev': base.tax_rev})
        
        #Add weighted contribution of each agent to expenditure and revenue
        for key in ['mod_ui_ex', 'mod_tax_rev', 'base_ui_ex', 'base_tax_rev']: 
            comb_dict[key]+= weight * agent_stats[key]
        
        #Add welfare contributions to combined stats
        dW_smooth_num += weight * dW_smooth
        dW_search_num += weight * dW_search
        
        dW_denom += weight * dW_mm

        #Add dict of stats for this agent type to be returned later                
        parts_dict.update({'type_' + str(i): copy.deepcopy(agent_stats)})
        
    #Calculate net welfare effects over all types    
    dW_smooth = dW_smooth_num / dW_denom
    dW_search = dW_search_num / dW_denom
    dW_tot = (dW_smooth_num + dW_search_num) / dW_denom
    
    comb_dict.update({'dW_smooth': dW_smooth,
                      'dW_search': dW_search,
                      'dW_tot' : dW_tot})
    
    return [parts_dict, comb_dict]



def baily_chetty(sim_agent, z_vals_base, z_vals_mod, eval_type,
                 cons_data_ui_norm = ui_cons_data_norm) :
    '''Baily Chetty calculation of welfare effects of benefit increase
    
    Note that these approximations calculate marginal utility of consumption
    from the realized consumption history supplied
    
    sim_agent:      agent_history_endo_js object w/ cons and emp histories
    z_vals base:    base income and benefit schedule
    z_vals_mod:     modified income and benefit schedule 
    eval_type:      Whether to evaluate benefit increase or extension
    '''
    #Agent parameters
    rho = sim_agent.p_d['rho']
    beta = sim_agent.p_d['beta_var']
    
    #Agent histories, and money-metric denominator
    c_hist = sim_agent.c_hist
    e_hist = sim_agent.e_hist
    
    #Marginal utility
    u = lambda c, rho=rho:  c**(1-rho)/(1-rho)
    uP = lambda c, rho=rho:  c**(-rho)
        
    #Money-metric denominator
    denom = u(1.01) - u(1.0)
    
    #Calculate time in each state; avg consumption and MU in each state from cons data
    c_data_ui = cons_data_ui_norm
    state_dict = {}
    for s in np.unique(e_hist):
        if s < len(c_data_ui): #Include only immediate post-exhaustion state
            share = np.mean(e_hist==s)
            c_avg = c_data_ui[s]
            MU = uP(c_avg)
            state_dict.update({s:{'share':share, 'c_avg':c_avg, 'MU':MU}})    
        
    
    #Welfare loss of changing taxes
    d_tax = z_vals_mod[0] - z_vals_base[0] #should be a negative number
    dW_tax = d_tax * state_dict[0]['MU'] * state_dict[0]['share']
    if dW_tax > 0:
        raise('welfare effect of raising taxes is positive!')
        
    #Welfare gain of increase:
    if eval_type == 'increase' or eval_type=='gruber':
        time_ui = 1 - np.mean(e_hist ==0) - np.mean(e_hist>7)
        d_inc = z_vals_mod[1] - z_vals_base[1] 
        
        if eval_type == 'gruber': #change in MU(c) based on 6.8% drop in cons
            cons_emp =  state_dict[0]['c_avg']
            cons_ui = (1 - 0.068) * cons_emp
            avg_MU = uP(cons_ui)
        
        elif eval_type =='increase': #Average MU(c) from simulated cons
            avg_MU = 0
            for s in range(1, 8):
                avg_MU += state_dict[s]['MU'] * state_dict[s]['share'] / time_ui
            
        dW_mod = time_ui * avg_MU * d_inc
     
    #Welfare gain of Extension:    
    elif eval_type == 'extension':
        time_extend = np.mean(e_hist==8)
        d_inc = z_vals_mod[8] - z_vals_base[8]
        avg_MU = state_dict[8]['MU']
        
        dW_mod = time_extend * avg_MU * d_inc
        
    if dW_mod < 0:
        raise('welfare effect of policy mod is negative!')
    
    #Return normalized welfare effects:
    dW_net = dW_tax + dW_mod
    dW_norm = dW_net / (u(1.01) - u(1.0)) #normalize by money metric
    return dW_net, dW_norm

def eval_welfare_nmh_rep(agent_pd, name, BCMC_dB = 1.0, BCMC_dT = 1.0 ):
    ############################################################################
    ################### No moral hazard##########
    ############################################################################
    pd_base = agent_pd
    
    #base case
    bs_nmh_base=run_sim(pd_base, z_vals_extend, z_exhaust, ex_search=True)
    
    #Cost and taxes
    nmh_dT_mc = (np.mean(bs_nmh_base.e_hist ==8)) * (z_vals_extend[7]-z_vals_extend[8])
    nmh_dT_bc = BCMC_dT * nmh_dT_mc #total behavioural cost of dT given some specified BCMC ratio
    nmh_dT_tax = nmh_dT_bc / np.mean(bs_nmh_base.e_hist==0) #tax on each employed period
    
    #Cost and Taxes for level increase
    nmh_dB_tax = nmh_dT_tax #Impose the same tax burden per employed period
    nmh_dB_rev = (nmh_dB_tax * np.mean(bs_nmh_base.e_hist==0)) / BCMC_dB #Collected revenue = total tax burden/(BC/MC)
    dB_nmh = nmh_dB_rev / (1
                          - np.mean(bs_nmh_base.e_hist==0)
                          - np.mean(bs_nmh_base.e_hist > 7)
                          )
    
    nmh_dT_tax_rate = (z_vals_extend[0]+nmh_dT_tax) / (z_vals_extend[0]) - 1
    nmh_dB_tax_rate = (z_vals_extend[0]+nmh_dB_tax) / (z_vals_extend[0]) - 1
    dB_nmh_rate = dB_nmh / z_vals_extend[1]
    
    #Benefit schedules
    z_dB_nmh=copy.deepcopy(z_vals_extend)
    for i in range(1,8):
        z_dB_nmh[i]=z_dB_nmh[i] + dB_nmh
    z_dB_nmh[0]= 1 - nmh_dB_tax
    
    z_dT_nmh = copy.deepcopy(z_vals_extend)
    z_dT_nmh[8]=z_dT_nmh[7]
    z_dT_nmh[0]=1 - nmh_dT_tax
    
    ########Buffer stock Simulations########
    bs_nmh_dB=run_sim(pd_base, z_dB_nmh, z_exhaust, ex_search=True)
    bs_nmh_dT=run_sim(pd_base, z_dT_nmh, z_exhaust, ex_search=True)
    bs_nmh_mm=run_sim(pd_base, z_mm, z_exhaust, ex_search=True)
    
    #Expenditure 
    (bs_nmh_dB.ui_ex - bs_nmh_base.ui_ex) -  bs_nmh_dB.tax_rev - bs_nmh_base.tax_rev
    (bs_nmh_dT.ui_ex - bs_nmh_base.ui_ex) -( bs_nmh_dT.tax_rev - bs_nmh_base.tax_rev)
    
    #welfare
    bs_nmh_dB_w = (bs_nmh_dB.u_tot - bs_nmh_base.u_tot) / (bs_nmh_mm.u_tot - bs_nmh_base.u_tot) 
    bs_nmh_dT_w = (bs_nmh_dT.u_tot - bs_nmh_base.u_tot) / (bs_nmh_mm.u_tot - bs_nmh_base.u_tot) 
    
    #Baily Chetty welfare
    nmh_dWdB_gruber = baily_chetty(bs_nmh_base, z_vals_extend, z_dB_nmh, 'gruber')[1] 
    nmh_dWdB_baily_chetty = baily_chetty(bs_nmh_base, z_vals_extend, z_dB_nmh, 'increase')[1]
    nmh_dWdT_baily_chetty = baily_chetty(bs_nmh_base, z_vals_extend, z_dT_nmh, 'extension')[1] 
    
    #Save model results
    out = {'name':name, 'search':'exogenous',
           'dB_rate':dB_nmh_rate, 'bcmc_dt':BCMC_dT, 'bcmc_db':BCMC_dB,
           'dWdB': bs_nmh_dB_w, 'dWdT':bs_nmh_dT_w,
           'tax_dB': 1-z_dB_nmh[0], 'tax_dT': 1-z_dT_nmh[0],
           'dWdB_gruber':nmh_dWdB_gruber,
           'dWdB_baily_chetty':nmh_dWdB_baily_chetty,
           'dWdT_baily_chetty':nmh_dWdT_baily_chetty,}
    
    #Consumption diagnostics - average consumption in each state
    e_hist = bs_nmh_base.e_hist
    c_hist = bs_nmh_base.c_hist
    for s in np.unique(e_hist):
        state_name = 'cbar_' + str(s)
        state_avg_cons = np.mean(c_hist[e_hist==s])
        out.update({state_name:state_avg_cons})
        
    #Consumption diagnostics - size of cons drop at onset
    onset_drops =np.array([])
    for index, state in np.ndenumerate(e_hist):
        if state ==1: #unemployment onset
            if index[0] > 0: #ignore agents that start unemployed
                index_lag = (index[0]-1, index[1])
                onset_drop =  c_hist[index_lag] - c_hist[index] 
                onset_drops = np.append(onset_drops,onset_drop)
    out.update({'onset_drop_25th': np.percentile(onset_drops, 25),
                'onset_drop_50th': np.percentile(onset_drops, 50),
                'onset_drop_75th': np.percentile(onset_drops, 75),})
    
    out_list.append(out)
        
        
def eval_welfare_nmh_het(mix_agent, name, BCMC_dB = 1.0, BCMC_dT = 1.0 ):        
    SS=mix_agent
    
    ###########Spender-save simulations############
    #base case
    bs_nmh_base=run_sim(pd_base, z_vals_extend, z_exhaust, ex_search=True)
    
    #Cost and taxes
    nmh_dT_mc = (np.mean(bs_nmh_base.e_hist ==8)) * (z_vals_extend[7]-z_vals_extend[8])
    nmh_dT_bc = BCMC_dT * nmh_dT_mc #total behavioural cost of dT given some specified BCMC ratio
    nmh_dT_tax = nmh_dT_bc / np.mean(bs_nmh_base.e_hist==0) #tax on each employed period
    
    #Cost and taxes for level increase
    nmh_dB_tax = nmh_dT_tax #Impose the same tax burden per employed period
    nmh_dB_rev = (nmh_dB_tax * np.mean(bs_nmh_base.e_hist==0)) / BCMC_dB #Collected revenue = total tax burden/(BC/MC)
    dB_nmh = nmh_dB_rev / (1
                          - np.mean(bs_nmh_base.e_hist==0)
                          - np.mean(bs_nmh_base.e_hist > 7)
                          )
    
    nmh_dT_tax_rate = (z_vals_extend[0]+nmh_dT_tax) / (z_vals_extend[0]) - 1
    nmh_dB_tax_rate = (z_vals_extend[0]+nmh_dB_tax) / (z_vals_extend[0]) - 1
    dB_nmh_rate = dB_nmh / z_vals_extend[1]
    
    #Benefit schedules
    z_dB_nmh=copy.deepcopy(z_vals_extend)
    for i in range(1,8):
        z_dB_nmh[i]=z_dB_nmh[i] + dB_nmh
    z_dB_nmh[0]= 1 - nmh_dB_tax
    
    z_dT_nmh = copy.deepcopy(z_vals_extend)
    z_dT_nmh[8]=z_dT_nmh[7]
    z_dT_nmh[0]=1 - nmh_dT_tax
    
    #Run Simulations
    ss_nmh_base = run_ss(SS, z_vals_extend, z_exhaust, ex_search=True)
    ss_nmh_mm = run_ss(SS, z_mm, z_exhaust, ex_search=True)
    ss_nmh_dB = run_ss(SS, z_dB_nmh, z_exhaust, ex_search=True)
    ss_nmh_dT = run_ss(SS, z_dT_nmh, z_exhaust, ex_search=True)
    
    #welfare
    ss_nmh_dB_stats= eval_ss(ss_nmh_dB, ss_nmh_base, ss_nmh_mm)
    ss_nmh_dT_stats= eval_ss(ss_nmh_dT, ss_nmh_base, ss_nmh_mm)
    
    #Taxes and expenditures
    ss_nmh_dB_stats[1]['mod_ui_ex']-ss_nmh_dB_stats[1]['base_ui_ex']
    ss_nmh_dB_stats[1]['mod_tax_rev']-ss_nmh_dB_stats[1]['base_tax_rev']
    ss_nmh_dT_stats[1]['mod_ui_ex']-ss_nmh_dB_stats[1]['base_ui_ex']
    ss_nmh_dT_stats[1]['mod_tax_rev']-ss_nmh_dB_stats[1]['base_tax_rev']
    
    #welfare
    ss_nmh_dB_w = ss_nmh_dB_stats[1]['dW_tot']
    ss_nmh_dB_smooth = ss_nmh_dB_stats[1]['dW_smooth']
    ss_nmh_dB_search = ss_nmh_dB_stats[1]['dW_search']
    
    ss_nmh_dT_w = ss_nmh_dT_stats[1]['dW_tot']
    ss_nmh_dT_smooth = ss_nmh_dT_stats[1]['dW_smooth']
    ss_nmh_dT_search = ss_nmh_dT_stats[1]['dW_search']
    
    
    out = {'name':name, 'search':'exogenous',
           'dB_rate':dB_nmh_rate, 'bcmc_dt':BCMC_dT, 'bcmc_db':BCMC_dB,
           'dWdB_cons': ss_nmh_dB_smooth, 'dWdT_cons':ss_nmh_dT_smooth,
           'dWdB_search': ss_nmh_dB_search, 'dWdT_search':ss_nmh_dT_search,
           'dWdB': ss_nmh_dB_w, 'dWdT':ss_nmh_dT_w,
           'tax_dB': 1-z_dB_nmh[0], 'tax_dT': 1-z_dT_nmh[0]}
    out_list.append(out)
    
        
def eval_welfare_endo_rep(agent_pd, name,  z0=None, dB=None): 
    ################### Moral Hazard generated by simulations###################
    pd_base = agent_pd
    if z0 == None and dB == None:
        #Compute the right tax andd benefit levels
        bs_endo_base = run_sim(pd_base, z_vals_extend, z_exhaust, ex_search=False)
        def find_z0_dT_obj(z0_dt):
            'objective function for finding post-tax employed income that finances extension'
            z_dT_endo = copy.deepcopy(z_vals_extend)
            z_dT_endo[8] = z_dT_endo[7] 
            z_dT_endo[0] = z0_dt
            
            bs_endo_dT = run_sim(pd_base, z_dT_endo, z_exhaust, ex_search=False)
            
            bs_endo_dT_d_ex = bs_endo_dT.ui_ex - bs_endo_base.ui_ex
            bs_endo_dT_d_rev = bs_endo_dT.tax_rev - bs_endo_base.tax_rev
            tax_gap = bs_endo_dT_d_ex - bs_endo_dT_d_rev
            
            return tax_gap
        
        def find_dB_obj(dB, z0=None):
            'objective function for finding level increase that balances tax'
            
            z_dB_endo = copy.deepcopy(z_vals_extend)
            z_dB_endo[0] = z0
            for i in range(1,8):
                z_dB_endo[i] = z_dB_endo[i] + dB
                
            bs_endo_dB = run_sim(pd_base, z_dB_endo, z_exhaust, ex_search=False)
                
            bs_endo_dB_d_ex = bs_endo_dB.ui_ex - bs_endo_base.ui_ex
            bs_endo_dB_d_rev = bs_endo_dB.tax_rev - bs_endo_base.tax_rev
            tax_gap = bs_endo_dB_d_ex - bs_endo_dB_d_rev
            
            return tax_gap
    
        #Find balanced budget tax and benefit
        z0_dT_endo = scipy.optimize.brentq(find_z0_dT_obj, 0.98, 1.0, maxiter =8,
                                           xtol=0.0005, rtol=0.000005)
        dB_endo = scipy.optimize.brentq(find_dB_obj, 0.000, 0.04, args=(z0_dT_endo),
                                          maxiter =8,  xtol=0.00005, rtol=0.00005)
    
    else:       
    #####to avoid having to recalculate####
        z0_dT_endo = z0
        dB_endo = dB
    
    
    print('post-tax income = {0:.4f}'.format(z0_dT_endo))
    print('Benefit increase = {0:.4f}'.format(dB_endo))
                
    #Set benefit schedules        
    z_dT_endo = copy.deepcopy(z_vals_extend)
    z_dT_endo[0] = z0_dT_endo
    z_dT_endo[8] = z_dT_endo[7] 
    
    z_dB_endo = copy.deepcopy(z_vals_extend)
    z_dB_endo[0]= z0_dT_endo
    dB_endo_rate = dB_endo/z_vals_extend[1]
    for i in range(1,8):
        z_dB_endo[i] = z_dB_endo[i] + dB_endo
    
    
    ###Buffer-stock Simulations###
    bs_endo_base = run_sim(pd_base, z_vals_extend, z_exhaust, ex_search=False)
    bs_endo_dB = run_sim(pd_base, z_dB_endo, z_exhaust, ex_search=False)
    bs_endo_dT = run_sim(pd_base, z_dT_endo, z_exhaust, ex_search=False)
    bs_endo_mm = run_sim(pd_base, z_mm, z_exhaust, ex_search=False)                 
    #
    
    #Expenditure and welfare
    bs_endo_dB_d_ex = bs_endo_dB.ui_ex - bs_endo_base.ui_ex
    bs_endo_dB_d_rev = bs_endo_dB.tax_rev - bs_endo_base.tax_rev
    bs_endo_dB_taxgap = bs_endo_dB_d_ex - bs_endo_dB_d_rev
    
    bs_endo_dT_d_ex = bs_endo_dT.ui_ex - bs_endo_base.ui_ex
    bs_endo_dT_d_rev = bs_endo_dT.tax_rev - bs_endo_base.tax_rev
    bs_endo_dT_taxgap = bs_endo_dT_d_ex - bs_endo_dT_d_rev
    
    print('dT tax gap = ' + str(bs_endo_dT_taxgap))
    print('dB tax gap = ' + str(bs_endo_dB_taxgap))
    
    
    #welfare
    bs_endo_dWdB = (bs_endo_dB.u_tot - bs_endo_base.u_tot) / (bs_endo_mm.u_tot - bs_endo_base.u_tot) 
    bs_endo_dWdT = (bs_endo_dT.u_tot - bs_endo_base.u_tot) / (bs_endo_mm.u_tot - bs_endo_base.u_tot) 
    
    #cons-smooth and search gains
    bs_endo_dWdB_cons = (bs_endo_dB.u_cons - bs_endo_base.u_cons) / (bs_endo_mm.u_tot - bs_endo_base.u_tot) 
    bs_endo_dWdB_search = (bs_endo_dB.u_search - bs_endo_base.u_search) / (bs_endo_mm.u_tot - bs_endo_base.u_tot) 
    
    bs_endo_dWdT_cons = (bs_endo_dT.u_cons - bs_endo_base.u_cons) / (bs_endo_mm.u_tot - bs_endo_base.u_tot) 
    bs_endo_dWdT_search = (bs_endo_dT.u_search - bs_endo_base.u_search) / (bs_endo_mm.u_tot - bs_endo_base.u_tot) 
    
    
    #BCMC
    bs_endo_mc_dB = (1- np.mean(bs_endo_base.e_hist==0) - np.mean(bs_endo_base.e_hist>7)) * dB_endo
    bs_endo_mc_dT = np.mean(bs_endo_base.e_hist==8)  * (z_vals_extend[7] - z_vals_extend[8])
    
    bs_endo_bc_dB = bs_endo_dB_d_ex
    bs_endo_bc_dT = bs_endo_dT_d_ex
    
    bcmc_bs_endo_dB = bs_endo_bc_dB/bs_endo_mc_dB
    bcmc_bs_endo_dT = bs_endo_bc_dT/bs_endo_mc_dT
    print('dB BC/MC ={0:.3f}'.format(bcmc_bs_endo_dB))
    print('dT BC/MC ={0:.3f}'.format(bcmc_bs_endo_dT))
    print('base, dB, dT avg_search = {0:.3f}, {1:.3f}, {2:.3f},'.format(bs_endo_base.avg_search, bs_endo_dB.avg_search, bs_endo_dT.avg_search))
    print('dWdB = {0:.5f}'.format(bs_endo_dWdB))
    print('dWdT = {0:.5f}'.format(bs_endo_dWdT))
    
    
    #Save result
    out = {'name':name, 'search':'endogenous',
           'dB_rate':dB_endo_rate,
           'bcmc_dt':bcmc_bs_endo_dT, 'bcmc_db':bcmc_bs_endo_dB,
           'dWdB': bs_endo_dWdB, 'dWdT':bs_endo_dWdT,
           'dWdB_cons': bs_endo_dWdB_cons, 'dWdT_cons':bs_endo_dWdT_cons,
           'dWdB_search': bs_endo_dWdB_search, 'dWdT_search':bs_endo_dWdT_search,
           'tax_dB': 1-z_dB_endo[0], 'tax_dT': 1-z_dT_endo[0],
           'tax_gap_dB':bs_endo_dB_taxgap, 'tax_gap_dT':bs_endo_dT_taxgap, }
    out_list.append(out)
     

def eval_welfare_endo_het(mix_agent, name,  z0=None, dB=None):   
    #########Spender-Saver Simulations########
    SS=mix_agent
    
    if z0==None and dB ==None:
    #    #Find tax rates and increases that balance tax and expenditure
        ss_endo_base = run_ss(SS, z_vals_extend, z_exhaust, ex_search=False)
        ss_endo_mm = run_ss(SS, z_mm, z_exhaust, ex_search=False)  
        def find_z0_dT_obj_ss(z0_dt):
            'objective function for finding post-tax employed income that finances extension'
            z_dT_endo_ss = copy.deepcopy(z_vals_extend)
            z_dT_endo_ss[8] = z_dT_endo_ss[7] 
            z_dT_endo_ss[0] = z0_dt
            
            ss_endo_dT = run_ss(SS, z_dT_endo_ss, z_exhaust, ex_search=False)      
            ss_endo_dT_stats = eval_ss(ss_endo_dT, ss_endo_base, ss_endo_mm)
            
            d_ex = ss_endo_dT_stats[1]['mod_ui_ex']-ss_endo_dT_stats[1]['base_ui_ex']
            d_tax = ss_endo_dT_stats[1]['mod_tax_rev']-ss_endo_dT_stats[1]['base_tax_rev']
            tax_gap = d_ex - d_tax
            return tax_gap
        
        def find_dB_obj_ss(dB, z0=None):
            'objective function for finding level increase that balances tax'
            
            z_dB_endo_ss = copy.deepcopy(z_vals_extend)
            z_dB_endo_ss[0] = z0
            for i in range(1,8):
                z_dB_endo_ss[i] = z_dB_endo_ss[i] + dB
                
            ss_endo_dB = run_ss(SS, z_dB_endo_ss, z_exhaust, ex_search=False)
            ss_endo_dB_stats = eval_ss(ss_endo_dB, ss_endo_base, ss_endo_mm)
            d_ex = ss_endo_dB_stats[1]['mod_ui_ex']-ss_endo_dB_stats[1]['base_ui_ex']
            d_tax = ss_endo_dB_stats[1]['mod_tax_rev']-ss_endo_dB_stats[1]['base_tax_rev']
            tax_gap = d_ex - d_tax
            return tax_gap
    #    
        #Find balanced budget tax and benefit
        z0_dT_endo_ss = scipy.optimize.brentq(find_z0_dT_obj_ss, 0.994, 1.0, maxiter=9, xtol=0.0005, rtol=0.0001)
        dB_endo_ss = scipy.optimize.brentq(find_dB_obj_ss, 0.005, 0.05, args= z0_dT_endo_ss,
                                           maxiter=9,  xtol=0.0005, rtol=0.0001)
        
    else:
        z0_dT_endo = z0
        dB_endo = dB
    print('post-tax income = {0:.4f}'.format(z0_dT_endo_ss))
    print('Benefit increase = {0:.4f}'.format(dB_endo_ss))
    
    #Set Benefit schedules
    z_dT_endo_ss = copy.deepcopy(z_vals_extend)
    z_dT_endo_ss[0] = z0_dT_endo_ss
    z_dT_endo_ss[8] = z_dT_endo_ss[7] 
      
    z_dB_endo_ss = copy.deepcopy(z_vals_extend)
    z_dB_endo_ss[0]=z0_dT_endo_ss
    dB_endo_rate_ss = dB_endo_ss/z_vals_extend[1]
    for i in range(1,8):
        z_dB_endo_ss[i] = z_dB_endo_ss[i] + dB_endo_ss
    
    #Run simulated histories
    ss_endo_base = run_ss(SS, z_vals_extend, z_exhaust, ex_search=False)
    ss_endo_mm = run_ss(SS, z_mm, z_exhaust, ex_search=False)  
    ss_endo_dB = run_ss(SS, z_dB_endo_ss, z_exhaust, ex_search=False)
    ss_endo_dT = run_ss(SS, z_dT_endo_ss, z_exhaust, ex_search=False)     
    
    ss_endo_dB_stats = eval_ss(ss_endo_dB, ss_endo_base, ss_endo_mm)
    ss_endo_dT_stats = eval_ss(ss_endo_dT, ss_endo_base, ss_endo_mm)
     
    #Expenditure and taxes
    ss_endo_dB_d_ex = ss_endo_dB_stats[1]['mod_ui_ex']-ss_endo_dB_stats[1]['base_ui_ex']
    ss_endo_dB_d_tax = ss_endo_dB_stats[1]['mod_tax_rev']-ss_endo_dB_stats[1]['base_tax_rev']
    ss_endo_dB_taxgap = ss_endo_dB_d_ex - ss_endo_dB_d_tax
    
    ss_endo_dT_d_ex = ss_endo_dT_stats[1]['mod_ui_ex']-ss_endo_dB_stats[1]['base_ui_ex']
    ss_endo_dT_d_tax = ss_endo_dT_stats[1]['mod_tax_rev']-ss_endo_dB_stats[1]['base_tax_rev']
    ss_endo_dT_taxgap = ss_endo_dT_d_ex - ss_endo_dT_d_tax
    
    print('dB tax gap = ' + str(ss_endo_dB_taxgap))
    print('dT tax gap = ' + str(ss_endo_dT_taxgap))
    
    #Welfare
    ss_endo_dB_w = ss_endo_dB_stats[1]['dW_tot']
    ss_endo_dT_w = ss_endo_dT_stats[1]['dW_tot']
    
    #Mechanical Costs
    ss_endo_mc_dB = 0.0
    ss_endo_mc_dT = 0.0
    for pair in ss_endo_base:
        weight = pair[0]
        agent = pair[1]
        
        mc_dB = (1- np.mean(agent.e_hist==0) - np.mean(agent.e_hist>7)) * dB_endo_ss
        mc_dT = np.mean(agent.e_hist==8) * (z_vals_extend[7] - z_vals_extend[8])
        
        ss_endo_mc_dB += weight*mc_dB
        ss_endo_mc_dT += weight*mc_dT
    
    #Behavioural cost
    ss_endo_bc_dB = ss_endo_dB_d_ex
    ss_endo_bc_dT = ss_endo_dT_d_ex
    
    #BCMC
    bcmc_ss_endo_dB = ss_endo_bc_dB / ss_endo_mc_dB 
    bcmc_ss_endo_dT = ss_endo_bc_dT / ss_endo_mc_dT 
    print('dB BC/MC ={0:.3f}'.format(bcmc_ss_endo_dB))
    print('dT BC/MC ={0:.3f}'.format(bcmc_ss_endo_dT))
    
    #Out
    out = {'name':name, 'search':'endogenous',
           'dB_rate':dB_endo_rate_ss,
           'bcmc_dt':bcmc_ss_endo_dT, 'bcmc_db':bcmc_ss_endo_dB, 
           'dWdB': ss_endo_dB_w, 'dWdT':ss_endo_dT_w,
           'tax_dB': 1-z_dB_endo_ss[0], 'tax_dT': 1-z_dB_endo_ss[0],
           'tax_gap_dB':ss_endo_dB_taxgap, 'tax_gap_dT':ss_endo_dT_taxgap }
    out_list.append(out)


###################################
#Main Simulation and Evaluation
###################################
out_list = []

BCMC_dB_bench = param.bcmc_db_svw
BCMC_dT_bench = param.bcmc_dt_svw

#Rep Agent    
eval_welfare_nmh_rep(pd_rep, name='rep_agent, no moral hazard')
eval_welfare_nmh_rep(pd_rep, name='rep_agent, calibrated moral hazard',
                     BCMC_dB = BCMC_dB_bench, BCMC_dT = BCMC_dT_bench)
eval_welfare_endo_rep(pd_rep, name='rep_agent, endogenous job search')
#
eval_welfare_nmh_rep(pd_gamma_1, name='rep_agent, gamma = 1, nmh')
eval_welfare_nmh_rep(pd_gamma_1, name='rep_agent, gamma = 1, calibrated mh',
                     BCMC_dB = BCMC_dB_bench, BCMC_dT = BCMC_dT_bench)
eval_welfare_endo_rep(pd_gamma_1, name='rep_agent gamma = 1, endogenous job search')
#
eval_welfare_nmh_rep(pd_gamma_4, name='rep_agent, gamma = 4, nmh')
eval_welfare_nmh_rep(pd_gamma_4, name='rep_agent, gamma = 4, calibrated mh',
                     BCMC_dB = BCMC_dB_bench, BCMC_dT = BCMC_dT_bench)
eval_welfare_endo_rep(pd_gamma_4, name='rep_agent gamma = 4, endogenous job search')
#
##Heterogeneity
eval_welfare_nmh_het(het_1b2k_agent, name='1b2k, no moral hazard')
eval_welfare_nmh_het(het_1b2k_agent, name='1b2k, calibrated moral hazard',
                     BCMC_dB = BCMC_dB_bench, BCMC_dT = BCMC_dT_bench)
eval_welfare_endo_het(het_1b2k_agent, name='1b2k, endogenous job search')
#
eval_welfare_nmh_het(het_2b2k_agent, name='2b2k, no moral hazard')
eval_welfare_nmh_het(het_2b2k_agent, name='2b2k, calibrated moral hazard',
                     BCMC_dB = BCMC_dB_bench, BCMC_dT = BCMC_dT_bench)
eval_welfare_endo_het(het_2b2k_agent, name='2b2k, endogenous job search')

fix_xi_key = '2b2k_xi = ' + str(models_params['est_params_2b2k_fix_xi']['phi'])
eval_welfare_nmh_het(het_2b2k_agent_fix_xi, name= fix_xi_key + ', no moral hazard')
eval_welfare_nmh_het(het_2b2k_agent_fix_xi, name=fix_xi_key + ', calibrated moral hazard',
                     BCMC_dB = BCMC_dB_bench, BCMC_dT = BCMC_dT_bench)
eval_welfare_endo_het(het_2b2k_agent_fix_xi, name=fix_xi_key + ', endogenous job search')

df = pd.DataFrame(out_list)
df.to_excel('../out/welfare_stats_log.xlsx')
    
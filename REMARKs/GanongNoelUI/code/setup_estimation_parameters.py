from __future__ import division
#from solve_cons import param_path
#TEMPORARY
import warnings



from time import time
import numpy as np
import pandas as pd
import warnings
import json
import csv


### JPMCI data ###
jpmc_param_path = "../Parameters/JPMC_inputs.json"
jpmc_infile = open(jpmc_param_path)
jpmc_params = json.load(jpmc_infile)
jpmc_infile.close

JPMC_cons_moments = jpmc_params['JPMC_cons_moments']
JPMC_search_moments = jpmc_params['JPMC_search_moments']
JPMC_cons_SE = jpmc_params['JPMC_cons_SE']
JPMC_search_SE = jpmc_params['JPMC_search_SE']

c_moments_len = jpmc_params['c_moments_len']
s_moments_len = jpmc_params['s_moments_len']
moments_len_diff = jpmc_params['moments_len_diff']

JPMC_cons_moments_FL = jpmc_params['JPMC_cons_moments_FL']
JPMC_search_moments_FL = jpmc_params['JPMC_search_moments_FL']
JPMC_cons_SE_FL = jpmc_params['JPMC_cons_SE_FL']
JPMC_search_SE_FL = jpmc_params['JPMC_search_SE_FL']

JPMC_cons_moments_ui = jpmc_params['JPMC_cons_moments_ui']
####################


### Estimation Params ###
param_path="../Parameters/params_ui.json"
infile = open(param_path)
scf_data_path = './'
json_params = json.load(infile)
infile.close()

# ------------------------------------------------------------------------------
# -------------- Set up decision problem and simulation parameters -------------
# ------------------------------------------------------------------------------
   
#begin Ganong code
Num_agents = int(json_params['Nagents'])
N_states = int(json_params['N_states'])
spline_k = int(json_params['spline_k'])
z_vals = np.array(json_params['z_vals'])
z_vals_FL = np.array(json_params['z_vals_FL'])
#z_vals_alt = np.array(json_params['z_vals_alt'])
Pi = np.array(json_params['Pi'])
Pi_extend = np.array(json_params['Pi_extend'])

beta= float(json_params['beta'])
beta_hyp= float(json_params['beta_hyp'])
rho=int(json_params['rho'])
TT = int(json_params['final_age']) - int(json_params['initial_age'])
    # In matlab:   param.final_age - param.initial_age    % 90-25; % Number of periods to iterate
    # int(json_params['total_simuation_periods_including_T'])  # Removed to comport with Matlab code
    # Total time: total periods including the final period.
Pi_t = [Pi] * TT

# Decision problem parameters:
R = float(json_params['R'])
R_save = float(json_params['R_save'])
R_bor = float(json_params['R_bor'])
L = float(json_params['L'])
a0_data=float(json_params['a0_data'])
constrained = bool(json_params['constrained'])

#job search parameters
phi = float(json_params['phi'])
k = float(json_params['k'])

#Parameters for plotting
e_extend = json_params['e_extend']
c_plt_start_index = json_params['c_plt_start_index']
s_plt_start_index = json_params['s_plt_start_index']
plt_norm_index = json_params['plt_norm_index']

#BCMC ratios from Schmieder and von Wachter (2016)
bcmc_db_svw = json_params['bcmc_db_svw']
bcmc_dt_svw = json_params['bcmc_dt_svw']


# ------------------------------------------------------------------------------
# ------------------------- Set Up the a-grid ----------------------------------
# ------------------------------------------------------------------------------

grid_type = json_params['grid_type']
exp_nest = int(json_params['exp_nest'])
a_min = float(json_params['a_min'])
a_max = float(json_params['a_max'])
a_size = int(json_params['a_size'])
a_extra = float(json_params['a_extra'])
a_huge = json_params['a_huge']
if a_huge is not None:
    a_huge = float(a_huge)    # TODO: we will need to figure out how to handle "None" in Matlab.


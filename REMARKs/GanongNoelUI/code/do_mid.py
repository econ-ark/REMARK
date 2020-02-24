# -*- coding: utf-8 -*-
"""
This script builds the model plots, calculates the standard errors for the
parameter estimates, and performs welfare simulations for the UI_spending paper.
It requires structural parameters to have been estimated and stored in:
    1. Parameters/model_params_main.json
    2. Parameters/model_params_sec.json
    3. Parameters/model_params_robust_gamma.json
These files are already found in this repository, but can be
re-estimated using the 'do_all.py' script, or manually using the
code-estimate_models.py script.

Plot output, model logfiles with goodness-of-fit measures and structural
parameters, parameter estimates' standard errors, and welfare statistics will
be (re)created in the out directory.
"""
import os
import sys
from time import time
from do_min import timestr, get_user_input


if __name__ == '__main__':
    
    prompt = '''Building plots, parameter standard errors, and performing welfare simulations takes about 45 minutes on a i7-7700 laptop with 16gb of RAM. Do you wish to proceed? Please enter 'y' or n' '''
    proceed = get_user_input(prompt)
    
    if proceed == 'n':
        exit()
    elif proceed == 'y':        
        print('Building plots...')
        sys.stdout.flush()
        ### Rebuild plots ###
        t_start = time()
        import model_plots
        import sparsity
        t_end = time()
        print('Building plots took' + timestr(t_end-t_start) + ' seconds.')
        sys.stdout.flush()
    
        ### Compute standard errors for parameter estimates ###
        print('Computing parameter standard errors...')
        sys.stdout.flush()
        t_start = time()
        import comp_SEs
        t_end = time()
        print('Computing standard errors for param. estimates took ' + timestr(t_end-t_start) + ' seconds.')
        sys.stdout.flush()
    
        ### Perform welfare simulations for structural models###
        print('Performing welfare simulations...')
        sys.stdout.flush()
        t_start = time()
        import model_welfare
        t_end = time()
        print('Welfare simulations took' + timestr(t_end-t_start) + ' seconds.')
        sys.stdout.flush()
    
    
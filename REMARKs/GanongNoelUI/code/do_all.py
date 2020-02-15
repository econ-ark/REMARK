# -*- coding: utf-8 -*-
"""
This script estimates the structural model parameters on consumption and job
search estimation targets from the JPMorgan Chase Insitute, builds the model
 plots from the estimated parameters, calculates standard errors for the
 parameter estimates, and performs welfare simulations.

The estimation targets are in the input/ directory, the estimated parameters will
be in the Parameters/ directory, and the plot output, model logfiles with
 goodness-of-fit measures, parameter estimates' standard errors, and welfare
 statistics will be in the out/ directory.
"""
import os
import sys
from time import time
from do_min import timestr, get_user_input


if __name__ == '__main__':
    
    prompt = '''Estimation can take up to 12 hours on a i7-7700 laptop with 16gb of RAM (to estimate models simultaneously or with different starting conditions, see estimate_models.py)Do you wish to proceed? Please enter 'y' or n' '''
    proceed = get_user_input(prompt)
    
    if proceed == 'n':
        exit()
    elif proceed == 'y':       
        
        ### Model estimation tagets ###
        print('Building model estimation targets...')
        sys.stdout.flush()
        t_start = time()
        import build_JPMC_targets
        t_end = time()
        print('Building model estimation targets took' + timestr(t_end-t_start) + ' seconds.')
        sys.stdout.flush()
              
        ### Estimate models ###
        print('Estimating models...')
        sys.stdout.flush()
        t_start = time()
        import estimate_models
        t_end = time()
        print('Estimating models took' + timestr(t_end-t_start) + ' seconds.')
        sys.stdout.flush()

        ### Estimate model robustness###
        print('Estimating models robustness...')
        sys.stdout.flush()
        t_start = time()
        import est_robust_gamma
        t_end = time()
        print('Estimating models robustness took' + timestr(t_end-t_start) + ' seconds.')
        sys.stdout.flush()

        ### Rebuild plots ###
        print('Building plots...')
        sys.stdout.flush()
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

    
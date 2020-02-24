# -*- coding: utf-8 -*-
"""
This script builds the model plots for the UI_spending paper. It requires
structural parameters to have been estimated and stored in:
    1. Parameters/model_params_main.json
    2. Parameters/model_params_sec.json
    3. Parameters/model_params_robust_gamma.json
These files are already found in this repository, but can be
re-estimated using the 'do_all.py' script, or manually using the
code-estimate_models.py script.

Plot output, and logfiles with goodness-of-fit measures and structural
parameters will be (re)created in the out directory.
"""

import os
import sys
from time import time

### Helper funcs ###
def timestr(seconds):
    return("{:.0f}".format(seconds))

def get_user_input(prompt):
    print(prompt)    
    sys.stdout.flush()
    while True:        
        user_input = raw_input()
        if user_input == 'y' or user_input == 'n':
            break
        else:
            print('''Bad input. Please enter 'y' or 'n' ''')
            sys.stdout.flush()
            continue
    return(user_input)



if __name__ == '__main__':
    prompt = '''Building plots takes about 10 minutes on a i7-7700 laptop with 16gb of RAM. Do you wish to proceed? Please enter 'y' or n' '''
    proceed = get_user_input(prompt)
    
    if proceed == 'n':
        exit()
    elif proceed == 'y':        
        print('building plots...')
        sys.stdout.flush()
        ### Rebuild plots ###
        t_start = time()
        import model_plots
        import sparsity
        t_end = time()
        print('Building plots took' + timestr(t_end-t_start) + ' seconds.')
        sys.stdout.flush()
        


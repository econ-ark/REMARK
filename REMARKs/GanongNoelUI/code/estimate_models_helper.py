# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:14:54 2018

@author: Xian_Work
"""


import json
import pandas as pd
import os

########################################################
#Read csv and make jsons of input dictionaries
########################################################
def grid_helper_in(opt_types = ['1b1k', '1b2k', '2b2k', '2d2k', 'fix_b1', 'fix_xi' ],
                   in_path = './est_models_in/',
                   out_path = './est_models_out/'):
    
    for opt_type in opt_types:
        f_name = './est_models_in/initial_conditions_' + opt_type + '.csv'
        df = pd.read_csv(f_name)
        for i in df.index:
            df.loc[i].to_json('./est_models_in/sim_' +  opt_type + '_{}.json'.format(i))
        
########################################################        
#Read output jsons and write to csv
########################################################
def grid_helper_out(opt_types = ['1b1k', '1b2k', '2b2k', '2d2k', 'fix_b1', 'fix_xi' ],
                   in_path = './est_models_in/',
                   out_path = './est_models_out/'):
            
            
    def get_files(start_str = " "):
        """Return files that start with start_str"""
        n = len(start_str)
        file_list = [f for f in os.listdir(out_path) if f[0:n] == start_str]
        return file_list
    
    #For each model optimized, dump results to a csv
    for opt_type in opt_types:
        out_path = './est_models_out/'
        start_str = "sim_" + opt_type    
        file_list = get_files(start_str)
        
        df = pd.DataFrame()
        i=0
        for f in file_list:
            path = out_path + f
            with open(path) as handle:
                dictdump = json.loads(handle.read())
                df_row = pd.DataFrame(dictdump, index=[i])
                df = df.append(df_row)
                i+=1
                
        df = df.sort_values('GOF')
        df.to_csv(out_path + 'est_' + opt_type + '.csv')
###
grid_helper_in()
grid_helper_out()

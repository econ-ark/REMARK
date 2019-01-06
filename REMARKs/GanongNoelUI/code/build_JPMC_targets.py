# -*- coding: utf-8 -*-
"""
Created on Tue Oct 02 10:10:15 2018

@author: Xian_Work
"""

import pandas as pd
import numpy as np
import json

out_dict = {}
jpmc_in = pd.read_excel('../input/gn_ui_targets2018-09-20.xls',
                        sheetname='tbl_model_targets')
jpmc_spend = jpmc_in.loc[jpmc_in['key'] == 'Spending']  


#########Consumption Moments##############
out_dict.update({'JPMC_cons_moments':np.round(jpmc_spend['value'], 3)})
out_dict.update({'JPMC_cons_moments_ui':np.round(jpmc_spend['value'], 3)[3:12]})

#Consumption Standard Errors
JPMC_cons_SE = np.round(jpmc_spend['se'], 6)
JPMC_cons_SE[0] = JPMC_cons_SE[1]
out_dict.update({'JPMC_cons_SE':JPMC_cons_SE})

out_dict.update({'c_moments_len':len(JPMC_cons_SE)})

#########Job-finding Hazard Moments##############
jpmc_in = pd.read_excel('../input/gn_ui_targets2018-09-20.xls',
                        sheetname='tbl_hazard_int')

JPMC_search_moments = list(np.round(jpmc_in['value'], 3))

moments_len_diff = out_dict['c_moments_len'] - len(JPMC_search_moments)
s_moments_len =  out_dict['c_moments_len'] - moments_len_diff

out_dict.update({'s_moments_len':s_moments_len})
out_dict.update({'moments_len_diff':moments_len_diff})

#Pad extra NAs
pad_list = ['--'] * moments_len_diff
JPMC_search_moments = pad_list + JPMC_search_moments
JPMC_search_moments = list(JPMC_search_moments)
out_dict.update({'JPMC_search_moments':JPMC_search_moments})

#Job-finding standard errors
JPMC_search_SE = list(np.round((jpmc_in['high'] - jpmc_in['low'])/3.92,5))
out_dict.update({'JPMC_search_SE':JPMC_search_SE})

####################################
#Florida moments
#####################################
jpmc_in = pd.read_excel('../input/gn_ui_targets2018-10-12.xls',
                        sheetname='tbl_pbd_stay_unemp')
FL_spend = jpmc_in.loc[jpmc_in['key'] == 'Spending']  
FL_spend = FL_spend.loc[FL_spend['pbd'] == '4 Months (Florida)']  

#Consumption moments
cons_moments_FL = list(np.round(FL_spend['value'],3))
cons_se_FL = list(np.round(FL_spend['se'],3))
cons_se_FL[0] = cons_se_FL[1] 

#Search moments taken out from JPMC by hand
search_moments_FL = ["--","--","--","--","--",0.3028,0.244,0.2972,0.3637,0.336,0.2377,0.4462,0.0343,0.389,0.1005,0.5973]
search_SE_FL = ["--","--","--","--","--", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

#Update
out_dict.update({"JPMC_cons_moments_FL":cons_moments_FL,
                 "JPMC_search_moments_FL":search_moments_FL,
                 "JPMC_cons_SE_FL":cons_se_FL,
                 "JPMC_search_SE_FL":search_SE_FL})

for k,v in out_dict.iteritems():
    if type(v) != int:
        out_dict.update({k:list(v)})


with open('../Parameters/JPMC_inputs.json', 'w') as f:
            json.dump(out_dict, f, indent=0)

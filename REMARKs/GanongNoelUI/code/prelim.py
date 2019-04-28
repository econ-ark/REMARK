# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:37:21 2017

@author: Xian_Work
"""

import os
import sys
import getpass
#from rpy2 import robjects
#import rpy2.robjects.lib.ggplot2 as ggplot2
#from rpy2.robjects.lib.ggplot2 import theme_bw
#from rpy2.robjects import pandas2ri
#pandas2ri.activate() 
#loc = robjects.r('c(0,0)')
#yrange_s = robjects.r('c(0.0,0.4)')
#yrange = robjects.r('c(0.75,1.02)')
#import make_plots as mp
#Import python modules
import numpy as np
import pandas as pd
from scipy.optimize import fmin, brute
import scipy
from time import time                                   # Used to time execution
import json
import csv
import copy

#Import parameters
import setup_estimation_parameters as param     #Import parameters 
T_series = 9                                    #Manually set number of periods


#import ui modules
import solve_cons
import solve_search


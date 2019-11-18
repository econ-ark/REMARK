## This should run all the code

import os

# Find pathname to this file:
my_file_path = os.path.dirname(os.path.abspath("do_all.py"))

# Change working directory to the one that has the code 
os.chdir(my_file_path + '/Code/Python')


import Template

# First, please follow "README.md" for all the preparation for running the code
# Setup
from dolo import *
import dolark 
from dolark import HModel # The model is written in yaml file, HModel is used to read the yaml file
from dolark.equilibrium import find_steady_state
from dolark.perturbation import perturb
from dolo import time_iteration, improved_time_iteration
from matplotlib import pyplot as plt
import numpy as np

#HModel reads the yaml file
aggmodel = HModel('Aiyagari.yaml')
aggmodel

# check features of the model
aggmodel.features 

eq = find_steady_state(aggmodel)
eq

#plot the wealth distribution
s = eq.dr.endo_grid.nodes() # grid for states (i.e the state variable--wealth in this case)
plt.plot(s, eq.μ.sum(axis=0), color='black')
plt.grid()
plt.title("Wealth Distribution")

# You can also check the steady state values of all variables
eq.as_df()

# define the dataframe
df = eq.as_df()

# plot relationship between assets of this period and of next period  
# altair plots a graph
import altair as alt
import pandas as pd

df = eq.as_df()
alt.Chart(df).mark_line().encode(
    x = alt.X('a', axis = alt.Axis(title='Current Assets')),
    y = alt.Y('i', axis=alt.Axis(title='Next Period Assets'))
)

# extract variables from the steady state solution
a = df['a']
r = df['r']
w = df['w']
e = df['e']
i = df['i']
μ = df['μ']

# generate a colume matrix with zeros
c = np.zeros((len(df),1))
income = np.zeros((len(df),1))
agg_c = 0
agg_inc = 0

# calculate consumption
for j in range(len(df)):
    c[j] = (1+r[j])*a[j] + w[j]*np.exp(e[j]) - i[j]

# calcuate income
for j in range(len(df)):
    income[j] = (r[j]+0.08)*a[j] + w[j]*np.exp(e[j])


# aggregate consumption and aggregat consumption
for j in range(len(df)):
    agg_c = agg_c + c[j]*μ[j]
    agg_inc = agg_inc + income[j]*μ[j]


saving = 1 - agg_c/agg_inc
saving

# Check the calibration of aggregate part of the model 
aggmodel.calibration      


# Check the calibration of the individual part of the model
aggmodel.model.calibration    

# Calculate the saving rate in the steady state under all kinds of calibrations
rows = []
rho_values = np.linspace(0, 0.9, 4)   #change serial correlation coefficent "rho "in {0, 0.3, 0.6, 0.9}
sig_values = np.linspace(0.2, 0.4, 2) #change the variance of labor shocks "sig" in {0.2, 0.4}
epsilon_values = np.linspace(1, 5, 3)       #change the coefficient of risk aversion {1,3,5}

for l in epsilon_values:
    aggmodel.model.set_calibration( epsilon = l)
    for n in sig_values:
        aggmodel.model.set_calibration( sig = n )
        for m in rho_values:
            aggmodel.model.set_calibration( rho=m )
            eq = find_steady_state(aggmodel)
            df = eq.as_df()
            a = df['a']
            r = df['r']
            w = df['w']
            e = df['e']
            μ = df['μ']
            i = df['i']
    
            #setup cnosumption and income
            c = np.zeros((len(df),1))
            income = np.zeros((len(df),1))
            agg_c = 0
            agg_inc = 0
        
            # calculate consumption
            for j in range(len(df)):
                c[j] = (1+r[j])*a[j] + w[j]*np.exp(e[j]) - i[j]
            # calcuate income
            for j in range(len(df)):
                income[j] = (r[j]+0.08)*a[j] + w[j]*np.exp(e[j])   #0.08 is the depreciation rate
            
            # aggregate consumption and aggregat consumption
            for j in range(len(df)):
                agg_c = agg_c + c[j]*μ[j]
                agg_inc = agg_inc + income[j]*μ[j]
            
            saving = (1 - agg_c/agg_inc)*100   #convert to %
            saving_rate = float("%.2f" % saving)  #with 2 decimals
            
            rows.append((l, n, m, saving_rate))
            
# import modules
import pandas as pd

# define df1 as the dataframe of saving rates I calcualted
df1 = pd.DataFrame(rows)

# change names of columns
df1.columns = ['Risk Averse Coefficient', 'Variance of Labor Shocks', 'Serial Correlation', 'Saving Rate']

# now I want to import data on saving rate calculated by Aiyagari(1994)
# Import the excel file and call it xls_file
xls_file = pd.ExcelFile('Aiyagari_SavingRate.xlsx')

# View the excel file's sheet names
xls_file.sheet_names

# Load the xls file's Sheet1 as a dataframe
# Place data on saving rates in Aiyagari(1994) into df2
df2 = xls_file.parse('Sheet1')

# Merge df1 and df2 and name it df3
df3 = pd.merge(df1,df2, on=['Risk Averse Coefficient','Variance of Labor Shocks', 'Serial Correlation'])

# Check the results
df3


# following steps tabulate the data frame
# first import tabulate
from tabulate import tabulate
# create the headers
headers = ["Risk Averse Coefficient", "Variance of Labor Shocks", "Serial Correlation", "Saving Rate","Saving Rate_Aiyagari"]

# create the markdown table
m = tabulate(df3,headers, tablefmt="github")
# open the markdown file
table = open("Table_SavingRate.md", "w")
#save the markdown table in the markdown file
table.write(m) 

# save it in a latex table
latex = tabulate(df3,headers, tablefmt="latex")
path = 'Tex\\Tables\\Table_SavingRate.tex'
table2 = open(path, 'w')
table2.write(latex)     #save the latex table

# plot wealth distribution under the baseline calibration
s = eq.dr.endo_grid.nodes() # grid for states (i.e the state variable--wealth in this case)
plt.plot(s[0:20], eq.μ.sum(axis=0)[0:20], color='black')   # I drop the last 10 grids when plotting since the probabilities of these levels of wealth are very close to zero. # The reason why I didn't use log for wealth is that taking log of a number which is extremely close to zero gets a very negative number 
plt.grid()
plt.title("Wealth Distribution")
plt.savefig('Figure_WealthDistribution.png')    # save the figure in the current directory 

# save the figure in the directory where TeX file is located.
save_results_to = 'Tex/Figures/'
plt.savefig(save_results_to + 'Figure_WealthDistribution.png', dpi = 300)

# Compile the LaTeX file
import subprocess
import os

FileDir = os.path.dirname(os.path.realpath('__file__'))
TexFile = os.path.join(FileDir, 'Tex/main.tex')
subprocess.check_call(['pdflatex', TexFile], cwd='Tex')
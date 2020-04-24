* AppendDataUsingSCF1992_2007.do
* This file gives selected varaibles of the Population 
clear 

cd $basePath/$logPath
cap log close
cap log using ./AppendDataUsingSCF1992_2007.log, replace
cd $basePath/$stataPath

** Construct data
do "SelectVarsUsingSCF1992.do"
do "SelectVarsUsingSCF1995.do"
do "SelectVarsUsingSCF1998.do"
do "SelectVarsUsingSCF2001.do"
do "SelectVarsUsingSCF2004.do"
do "SelectVarsUsingSCF2007.do"

cd ../../Data/Constructed
append using SCF2004_population
append using SCF2001_population
append using SCF1998_population
append using SCF1995_population
append using SCF1992_population
drop if AGE<26
drop if AGE>65

** Save data
save SCF1992_2007_population, replace

/* Note: In the waves between 1995 and 2004, levels of normal income are reported. I interpret the level of normal income as being permanent income. Levels of normal income are not reported in the 1992 wave. Instead, in this wave there is a variable which reports whether the level of income is normal or not. Regarding the 1992 wave, only observations which report that the level of income is normal are used, and the levels of income of remaining observations in the 1992 wave are interpreted as the levels of permanent income. */

cd $basePath/$stataPath /* When program ends, make sure working directory is the program's directory */

log close

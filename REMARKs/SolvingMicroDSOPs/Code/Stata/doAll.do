* Assuming existence of all 'raw' data files in appropriate locations, create processed files containing needed data

set linesize 200

* make paths
global basePath  "/Volumes/Data/Notes/NumericalMethods/SolvingMicroDSOPs/Latest"
global stataPath "Code/Stata"
global logPath   "Code/Stata" 

cd $basePath/$stataPath

do AppendDataUsingSCF1992_2007.do
do WIRatioPopulation.do


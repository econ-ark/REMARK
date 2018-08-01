
Sticky Expectations and Consumption Dynamics--Empirical Archive
===============================================================
Carroll, Crawley, Slacalek, Tokuoka and White, February 5, 2018,
jiri.slacalek@gmail.com

This archive replicates empirical results reported in the paper.
All programs are written in Stata 15.0

Figure 1: habitsHistogram.do in folder "./Code/Empirical/metaAnalysis"
            The data for the figure are from Havranek et al. (EER, 2017); http://meta-analysis.cz/habits/; file habit.dta

Table 3: usConsDynEmp.do in folder "./Code/Empirical"
            folder "./Code/Empirical/data" contains the Excel dataset with US data (usData_20170927.xls)
            folder "./Code/Empirical/docs" contains LaTeX file tableTemplate.tex (which can be used to produce a pdf with regression results)
Notes
-----
The do files are called by the MAIN Python file in ./Code/Models if the appropriate flags are set.
To run the do files independently, the user should pass an argument to the Stata do call with the root
directory of this archive.

For questions and comments, email jiri.slacalek@gmail.com
	



In order for the files in this directory to work properly, the Federal
Reserve's SCF datasets 
that the programs use must be located in
appropriate directories that are accessible to the 
programs.  For
example, the 1992 SCF needs to be located at

   

../../../Downloads/SCF/1992



An all the required SCF datasets in the appropriate directory
structure can be downloaded from

   

ftp://llorracc.net/VaultPub/Data/SCF



or the entire set of SCF's (warning: it is very large) from


   
   ftp://llorracc.net/VaultPub/Data/SCF.zip



which has SCF files downloaded on 2011/08/06. 

Alternatively, the latest versions of the individual 
files can be
obtained directly from the Fed's website:

   

http://www.federalreserve.gov/pubs/oss/oss2/scfindex.html



but if you download them one-by-one you need to make sure that you put them in a directory structure 
corresponding 
to the structure at 

ftp://llorracc.net/VaultPub/Data/SCF


(you don't need to download the extra files like codebooks etc; all that is needed for the programs to 
work is the
Stata scf files, like scf92.dta for the 1992 SCF, which should be in a directory 1992/scf92.dta)

***********************************************************************************************************

doAll.do file runs all the programs.

In Particular:

1) SelectVarsUsingSCFXXXX.do: Selects the variables from the SCF raw data and construct the Permanent income, 
                              wealth and the weights of each household in the population for the year XXXX.
2) AppendDataUsingSCF1992_2007.do: Appends the outcomes of the SelectVarsUsingSCFXXXX.
3) WIRatioPopulation.do: Constructs the Wealth to after tax permanent income ratio of each households. And save the 
                         output "SCFdata.txt" in the folder "./Code/Mathematica/StructuralEstimation" which 
                         is used by the Mathematica programs to estimate the structural parameters.


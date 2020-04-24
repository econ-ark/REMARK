/* This program gives the Summary statistics for Income, Net Worth and 
Wealth/Income Ratio of the Married Households whose ages are between 
31 and 55. 
The program builts on the results obtained by doAll.do file, so run this
file after running the doAll.do file.
*/

cd $basePath/$stataPath

cd ../../Data/Constructed

***************************************************************************************************
/* Specifies the list of percentiles of INCOME, NETW and WIRATIO. p50 represents 50th percentile: median. 
   Modify the list if you want to obtain results for different percentiles.*/
global percentiles = "p50"       

scalar AgeRange1 = `"26-30"'   
scalar AgeRange2 = `"31-35"'    
scalar AgeRange3 = `"36-40"'
scalar AgeRange4 = `"41-45"'
scalar AgeRange5 = `"46-50"'
scalar AgeRange6 = `"51-55"'
scalar AgeRange7 = `"56-60"'

***************************************************************************************************
*************************************  1992 Survey Summary ******************************************

use $basePath/Data/Constructed/SCF1992_2007_population, clear
keep if YEAR == 1992
xtset HHID
sort HHID
by HHID: gen OBS=_n
xtset HHID OBS

egen AVGWGT = sum(WGT), by(HHID)          /* Generates one Weight for each Household */
egen AVGINC = sum(INCOME*WGT), by(HHID)   /* This line and the following line Generate the Average Income for each Household */
replace AVGINC = AVGINC/AVGWGT

egen AVGNETW = sum(NETW*WGT), by(HHID)     /* This line and the following line Generate the Average Net Worth for each Household */
replace AVGNETW = AVGNETW/AVGWGT

gen AVGWIRATIO = AVGNETW/AVGINC          /* Generates the Average Wealth/Income Ratio for each Household */

************************************** Age Range Selection *****************************

xtsum HHID
keep if OBS==1
drop OBS 

keep if  AGE >= 26 & AGE <= 60       /* Constructs 5 year period age groups: 31-35, ...., 51-55. */
gen AGEID = int((AGE-26)/5)+1
xtset  AGEID
sort AGEID HHID

***************** Before Tax Permanent Income / After Tax Permanent Income RATIO *****************

/* This section gives the ratio: Before Tax Permanent Income / After Tax Permanent Income
  This adjustment is necessary; we need to rescale WIRATIO properly, since WIRATIO obtained 
 using STATA is the ratio of wealth to before tax permanent income, not to after tax permanent income.
 (Note that the work in the MICRODSOP lecture notes takes parameters from Cagetti (2003)
 which is based on after tax income.)
 */

 /*  Income and IncomeRatio are calculated using data in Cagetti (2003) and SCF data  */
 matrix input RawMat = (1.1758, 39497 \ 1.221, 49262 \ 1.2874, 61057 \ 1.2594, 68224 \ 1.4432, 86353 ///
                     \ 1.5055, 96983 \ 1.5509, 98786 \ 1.5663, 1.0223e+005 \ 1.5663, 1e+010 ) 
svmat RawMat
rename RawMat1 RAWIRATIO
rename RawMat2 RAWI	

gen TXIRATIO =.    /* := Before Tax Permanent Income / After Tax Permanent Income */         
local N=_N

qui forvalues i=1/`N' {                                  /* Gives the Before Tax Permanent Income / After Tax Permanent Income RATIO */
 		   replace RAWI = AVGINC[`i'] in 10
           ipolate RAWIRATIO RAWI, gen(TEMP) epolate
		   replace TXIRATIO = TEMP[10] in `i'
		   drop TEMP 
		  }
replace RAWI =. in 10
replace TXIRATIO = 1 if TXIRATIO < 1

/* AfterTax adjustment of Wealth/Income Ratio: AVGWIRATIO represents Net Worth/Before Tax Permanent Income for each HH, 
thus multiplying AVGWIRATIO by TXIRATIO (Bef. Tax Inc / After Tax Inc) gives AVGWIRATIO= Net Worth/After Tax Inc, which is desired.  */	

replace AVGWIRATIO = AVGWIRATIO*TXIRATIO	
  	  
**************************************************************************************************		  

qui: sum AGEID, d
local size=r(max)                        /* size is used as an index number in the following loops*/

gen AVGINCBYAGE = .
gen OBSBYAGE = .

qui foreach p in $percentiles {
  gen `p'INCBYAGE = .
   qui forvalues k = 1/`size' {                                   /* Generates the Average Income for each Age Group defined by AGEID */
           sum AVGINC [aweight = AVGWGT] if AGEID==`k', d     
           replace AVGINCBYAGE = r(mean) if AGEID==`k'
           replace `p'INCBYAGE = r(`p') if AGEID==`k'
		   replace OBSBYAGE = r(N) if AGEID==`k'
		  }
		} 
		
gen AVGNETWBYAGE = .

qui foreach p in $percentiles {
  gen `p'NETWBYAGE = .
  qui forvalues k = 1/`size' {                                   /* Generates the Net Worth for each Age Group defined by AGEID */
           sum AVGNETW [aweight = AVGWGT] if AGEID==`k', d 
           replace AVGNETWBYAGE = r(mean) if AGEID==`k'
           replace `p'NETWBYAGE = r(`p') if AGEID==`k'
		  }		  
		 } 
		 
gen AVGWIRATIOBYAGE = .

qui foreach p in $percentiles {
  gen `p'WIRATIOBYAGE = .
  qui forvalues k = 1/`size' {                                   /* Generates the Wealth/Income Ratio for each Age Group defined by AGEID */
           sum AVGWIRATIO [aweight = AVGWGT] if AGEID==`k', d 
           replace AVGWIRATIOBYAGE = r(mean) if AGEID==`k'
           replace `p'WIRATIOBYAGE = r(`p') if AGEID==`k'
		  }
		 }

gen AGERANGE= `""'
qui forvalues k=1/`size' {
           replace AGERANGE = AgeRange`k' if AGEID==`k'        /* Generates string values for each Age Group which are used in graphs for illustrative purposes */
          }

by AGEID: gen OBS=_n
save "$basePath/Data/Constructed/All1992", replace 

*************************************  1995 Survey Summary ******************************************

use $basePath/Data/Constructed/SCF1992_2007_population, clear
keep if YEAR == 1995

xtset HHID
sort HHID
by HHID: gen OBS=_n
xtset HHID OBS

egen AVGWGT = sum(WGT), by(HHID)          /* Generates one Weight for each Household */
egen AVGINC = sum(INCOME*WGT), by(HHID)   /* This line and the following line Generate the Average Income for each Household */
replace AVGINC = AVGINC/AVGWGT

egen AVGNETW = sum(NETW*WGT), by(HHID)     /* This line and the following line Generate the Average Net Worth for each Household */
replace AVGNETW = AVGNETW/AVGWGT

gen AVGWIRATIO = AVGNETW/AVGINC          /* Generates the Average Wealth/Income Ratio for each Household */

************************************** Age Range Selection *****************************

xtsum HHID
keep if OBS==1
drop OBS 

keep if  AGE >= 26 & AGE <= 60       /* Constructs 5 year period age groups: 31-35, ...., 51-55. */
gen AGEID = int((AGE-26)/5)+1
xtset  AGEID
sort AGEID HHID

***************** Before Tax Permanent Income / After Tax Permanent Income RATIO *****************

/* This section gives the ratio: Before Tax Permanent Income / After Tax Permanent Income
  This adjustment is necessary; we need to rescale WIRATIO properly, since WIRATIO obtained 
 using STATA is the ratio of wealth to before tax permanent income, not to after tax permanent income.
 (Note that the work in the MICRODSOP lecture notes takes parameters from Cagetti (2003)
 which is based on after tax income.)
 */

 /*  Income and IncomeRatio are calculated using data in Cagetti (2003) and SCF data  */
 matrix input RawMat = (1.1758, 39497 \ 1.221, 49262 \ 1.2874, 61057 \ 1.2594, 68224 \ 1.4432, 86353 ///
                     \ 1.5055, 96983 \ 1.5509, 98786 \ 1.5663, 1.0223e+005 \ 1.5663, 1e+010 ) 
svmat RawMat
rename RawMat1 RAWIRATIO
rename RawMat2 RAWI	

gen TXIRATIO =.    /* := Before Tax Permanent Income / After Tax Permanent Income */         
local N=_N

qui forvalues i=1/`N' {                                  /* Gives the Before Tax Permanent Income / After Tax Permanent Income RATIO */
 		   replace RAWI = AVGINC[`i'] in 10
           ipolate RAWIRATIO RAWI, gen(TEMP) epolate
		   replace TXIRATIO = TEMP[10] in `i'
		   drop TEMP 
		  }
replace RAWI =. in 10
replace TXIRATIO = 1 if TXIRATIO < 1

/* AfterTax adjustment of Wealth/Income Ratio: AVGWIRATIO represents Net Worth/Before Tax Permanent Income for each HH, 
thus multiplying AVGWIRATIO by TXIRATIO (Bef. Tax Inc / After Tax Inc) gives AVGWIRATIO= Net Worth/After Tax Inc, which is desired.  */	

replace AVGWIRATIO = AVGWIRATIO*TXIRATIO	
  	  
**************************************************************************************************		  

qui: sum AGEID, d
local size=r(max)                        /* size is used as an index number in the following loops*/

gen AVGINCBYAGE = .
gen OBSBYAGE = .

qui foreach p in $percentiles {
  gen `p'INCBYAGE = .
   qui forvalues k = 1/`size' {                                   /* Generates the Average Income for each Age Group defined by AGEID */
           sum AVGINC [aweight = AVGWGT] if AGEID==`k', d     
           replace AVGINCBYAGE = r(mean) if AGEID==`k'
           replace `p'INCBYAGE = r(`p') if AGEID==`k'
		   replace OBSBYAGE = r(N) if AGEID==`k'
		  }
		} 
		
gen AVGNETWBYAGE = .

qui foreach p in $percentiles {
  gen `p'NETWBYAGE = .
  qui forvalues k = 1/`size' {                                   /* Generates the Net Worth for each Age Group defined by AGEID */
           sum AVGNETW [aweight = AVGWGT] if AGEID==`k', d 
           replace AVGNETWBYAGE = r(mean) if AGEID==`k'
           replace `p'NETWBYAGE = r(`p') if AGEID==`k'
		  }		  
		 } 
		 
gen AVGWIRATIOBYAGE = .

qui foreach p in $percentiles {
  gen `p'WIRATIOBYAGE = .
  qui forvalues k = 1/`size' {                                   /* Generates the Wealth/Income Ratio for each Age Group defined by AGEID */
           sum AVGWIRATIO [aweight = AVGWGT] if AGEID==`k', d 
           replace AVGWIRATIOBYAGE = r(mean) if AGEID==`k'
           replace `p'WIRATIOBYAGE = r(`p') if AGEID==`k'
		  }
		 }

gen AGERANGE= `""'
qui forvalues k=1/`size' {
           replace AGERANGE = AgeRange`k' if AGEID==`k'        /* Generates string values for each Age Group which are used in graphs for illustrative purposes */
          }

by AGEID: gen OBS=_n
save "$basePath/Data/Constructed/All1995", replace 

*************************************  1998 Survey Summary ******************************************

use $basePath/Data/Constructed/SCF1992_2007_population, clear
keep if YEAR == 1998

xtset HHID
sort HHID
by HHID: gen OBS=_n
xtset HHID OBS

egen AVGWGT = sum(WGT), by(HHID)          /* Generates one Weight for each Household */
egen AVGINC = sum(INCOME*WGT), by(HHID)   /* This line and the following line Generate the Average Income for each Household */
replace AVGINC = AVGINC/AVGWGT

egen AVGNETW = sum(NETW*WGT), by(HHID)     /* This line and the following line Generate the Average Net Worth for each Household */
replace AVGNETW = AVGNETW/AVGWGT

gen AVGWIRATIO = AVGNETW/AVGINC          /* Generates the Average Wealth/Income Ratio for each Household */

************************************** Age Range Selection *****************************

xtsum HHID
keep if OBS==1
drop OBS 

keep if  AGE >= 26 & AGE <= 60       /* Constructs 5 year period age groups: 31-35, ...., 51-55. */
gen AGEID = int((AGE-26)/5)+1
xtset  AGEID
sort AGEID HHID

***************** Before Tax Permanent Income / After Tax Permanent Income RATIO *****************

/* This section gives the ratio: Before Tax Permanent Income / After Tax Permanent Income
  This adjustment is necessary; we need to rescale WIRATIO properly, since WIRATIO obtained 
 using STATA is the ratio of wealth to before tax permanent income, not to after tax permanent income.
 (Note that the work in the MICRODSOP lecture notes takes parameters from Cagetti (2003)
 which is based on after tax income.)
 */

 /*  Income and IncomeRatio are calculated using data in Cagetti (2003) and SCF data  */
 matrix input RawMat = (1.1758, 39497 \ 1.221, 49262 \ 1.2874, 61057 \ 1.2594, 68224 \ 1.4432, 86353 ///
                     \ 1.5055, 96983 \ 1.5509, 98786 \ 1.5663, 1.0223e+005 \ 1.5663, 1e+010 ) 
svmat RawMat
rename RawMat1 RAWIRATIO
rename RawMat2 RAWI	

gen TXIRATIO =.    /* := Before Tax Permanent Income / After Tax Permanent Income */         
local N=_N

qui forvalues i=1/`N' {                                  /* Gives the Before Tax Permanent Income / After Tax Permanent Income RATIO */
 		   replace RAWI = AVGINC[`i'] in 10
           ipolate RAWIRATIO RAWI, gen(TEMP) epolate
		   replace TXIRATIO = TEMP[10] in `i'
		   drop TEMP 
		  }
replace RAWI =. in 10
replace TXIRATIO = 1 if TXIRATIO < 1

/* AfterTax adjustment of Wealth/Income Ratio: AVGWIRATIO represents Net Worth/Before Tax Permanent Income for each HH, 
thus multiplying AVGWIRATIO by TXIRATIO (Bef. Tax Inc / After Tax Inc) gives AVGWIRATIO= Net Worth/After Tax Inc, which is desired.  */	

replace AVGWIRATIO = AVGWIRATIO*TXIRATIO	
  	  
**************************************************************************************************		  

qui: sum AGEID, d
local size=r(max)                        /* size is used as an index number in the following loops*/

gen AVGINCBYAGE = .
gen OBSBYAGE = .

qui foreach p in $percentiles {
  gen `p'INCBYAGE = .
   qui forvalues k = 1/`size' {                                   /* Generates the Average Income for each Age Group defined by AGEID */
           sum AVGINC [aweight = AVGWGT] if AGEID==`k', d     
           replace AVGINCBYAGE = r(mean) if AGEID==`k'
           replace `p'INCBYAGE = r(`p') if AGEID==`k'
		   replace OBSBYAGE = r(N) if AGEID==`k'
		  }
		} 
		
gen AVGNETWBYAGE = .

qui foreach p in $percentiles {
  gen `p'NETWBYAGE = .
  qui forvalues k = 1/`size' {                                   /* Generates the Net Worth for each Age Group defined by AGEID */
           sum AVGNETW [aweight = AVGWGT] if AGEID==`k', d 
           replace AVGNETWBYAGE = r(mean) if AGEID==`k'
           replace `p'NETWBYAGE = r(`p') if AGEID==`k'
		  }		  
		 } 
		 
gen AVGWIRATIOBYAGE = .

qui foreach p in $percentiles {
  gen `p'WIRATIOBYAGE = .
  qui forvalues k = 1/`size' {                                   /* Generates the Wealth/Income Ratio for each Age Group defined by AGEID */
           sum AVGWIRATIO [aweight = AVGWGT] if AGEID==`k', d 
           replace AVGWIRATIOBYAGE = r(mean) if AGEID==`k'
           replace `p'WIRATIOBYAGE = r(`p') if AGEID==`k'
		  }
		 }

gen AGERANGE= `""'
qui forvalues k=1/`size' {
           replace AGERANGE = AgeRange`k' if AGEID==`k'        /* Generates string values for each Age Group which are used in graphs for illustrative purposes */
          }

by AGEID: gen OBS=_n
save "$basePath/Data/Constructed/All1998", replace 

*************************************  2001 Survey Summary ******************************************

use $basePath/Data/Constructed/SCF1992_2007_population, clear
keep if YEAR == 2001

xtset HHID
sort HHID
by HHID: gen OBS=_n
xtset HHID OBS

egen AVGWGT = sum(WGT), by(HHID)          /* Generates one Weight for each Household */
egen AVGINC = sum(INCOME*WGT), by(HHID)   /* This line and the following line Generate the Average Income for each Household */
replace AVGINC = AVGINC/AVGWGT

egen AVGNETW = sum(NETW*WGT), by(HHID)     /* This line and the following line Generate the Average Net Worth for each Household */
replace AVGNETW = AVGNETW/AVGWGT

gen AVGWIRATIO = AVGNETW/AVGINC          /* Generates the Average Wealth/Income Ratio for each Household */

************************************** Age Range Selection *****************************

xtsum HHID
keep if OBS==1
drop OBS 

keep if  AGE >= 26 & AGE <= 60       /* Constructs 5 year period age groups: 31-35, ...., 51-55. */
gen AGEID = int((AGE-26)/5)+1
xtset  AGEID
sort AGEID HHID

***************** Before Tax Permanent Income / After Tax Permanent Income RATIO *****************

/* This section gives the ratio: Before Tax Permanent Income / After Tax Permanent Income
  This adjustment is necessary; we need to rescale WIRATIO properly, since WIRATIO obtained 
 using STATA is the ratio of wealth to before tax permanent income, not to after tax permanent income.
 (Note that the work in the MICRODSOP lecture notes takes parameters from Cagetti (2003)
 which is based on after tax income.)
 */

 /*  Income and IncomeRatio are calculated using data in Cagetti (2003) and SCF data  */
 matrix input RawMat = (1.1758, 39497 \ 1.221, 49262 \ 1.2874, 61057 \ 1.2594, 68224 \ 1.4432, 86353 ///
                     \ 1.5055, 96983 \ 1.5509, 98786 \ 1.5663, 1.0223e+005 \ 1.5663, 1e+010 ) 
svmat RawMat
rename RawMat1 RAWIRATIO
rename RawMat2 RAWI	

gen TXIRATIO =.    /* := Before Tax Permanent Income / After Tax Permanent Income */         
local N=_N

qui forvalues i=1/`N' {                                  /* Gives the Before Tax Permanent Income / After Tax Permanent Income RATIO */
 		   replace RAWI = AVGINC[`i'] in 10
           ipolate RAWIRATIO RAWI, gen(TEMP) epolate
		   replace TXIRATIO = TEMP[10] in `i'
		   drop TEMP 
		  }
replace RAWI =. in 10
replace TXIRATIO = 1 if TXIRATIO < 1

/* AfterTax adjustment of Wealth/Income Ratio: AVGWIRATIO represents Net Worth/Before Tax Permanent Income for each HH, 
thus multiplying AVGWIRATIO by TXIRATIO (Bef. Tax Inc / After Tax Inc) gives AVGWIRATIO= Net Worth/After Tax Inc, which is desired.  */	

replace AVGWIRATIO = AVGWIRATIO*TXIRATIO	
  	  
**************************************************************************************************		  

qui: sum AGEID, d
local size=r(max)                        /* size is used as an index number in the following loops*/

gen AVGINCBYAGE = .
gen OBSBYAGE = .

qui foreach p in $percentiles {
  gen `p'INCBYAGE = .
   qui forvalues k = 1/`size' {                                   /* Generates the Average Income for each Age Group defined by AGEID */
           sum AVGINC [aweight = AVGWGT] if AGEID==`k', d     
           replace AVGINCBYAGE = r(mean) if AGEID==`k'
           replace `p'INCBYAGE = r(`p') if AGEID==`k'
		   replace OBSBYAGE = r(N) if AGEID==`k'
		  }
		} 
		
gen AVGNETWBYAGE = .

qui foreach p in $percentiles {
  gen `p'NETWBYAGE = .
  qui forvalues k = 1/`size' {                                   /* Generates the Net Worth for each Age Group defined by AGEID */
           sum AVGNETW [aweight = AVGWGT] if AGEID==`k', d 
           replace AVGNETWBYAGE = r(mean) if AGEID==`k'
           replace `p'NETWBYAGE = r(`p') if AGEID==`k'
		  }		  
		 } 
		 
gen AVGWIRATIOBYAGE = .

qui foreach p in $percentiles {
  gen `p'WIRATIOBYAGE = .
  qui forvalues k = 1/`size' {                                   /* Generates the Wealth/Income Ratio for each Age Group defined by AGEID */
           sum AVGWIRATIO [aweight = AVGWGT] if AGEID==`k', d 
           replace AVGWIRATIOBYAGE = r(mean) if AGEID==`k'
           replace `p'WIRATIOBYAGE = r(`p') if AGEID==`k'
		  }
		 }

gen AGERANGE= `""'
qui forvalues k=1/`size' {
           replace AGERANGE = AgeRange`k' if AGEID==`k'        /* Generates string values for each Age Group which are used in graphs for illustrative purposes */
          }

by AGEID: gen OBS=_n
save "$basePath/Data/Constructed/All2001", replace 

*************************************  2004 Survey Summary ******************************************

use $basePath/Data/Constructed/SCF1992_2007_population, clear
keep if YEAR == 2004

xtset HHID
sort HHID
by HHID: gen OBS=_n
xtset HHID OBS

egen AVGWGT = sum(WGT), by(HHID)          /* Generates one Weight for each Household */
egen AVGINC = sum(INCOME*WGT), by(HHID)   /* This line and the following line Generate the Average Income for each Household */
replace AVGINC = AVGINC/AVGWGT

egen AVGNETW = sum(NETW*WGT), by(HHID)     /* This line and the following line Generate the Average Net Worth for each Household */
replace AVGNETW = AVGNETW/AVGWGT

gen AVGWIRATIO = AVGNETW/AVGINC          /* Generates the Average Wealth/Income Ratio for each Household */

************************************** Age Range Selection *****************************

xtsum HHID
keep if OBS==1
drop OBS 

keep if  AGE >= 26 & AGE <= 60       /* Constructs 5 year period age groups: 31-35, ...., 51-55. */
gen AGEID = int((AGE-26)/5)+1
xtset  AGEID
sort AGEID HHID

***************** Before Tax Permanent Income / After Tax Permanent Income RATIO *****************

/* This section gives the ratio: Before Tax Permanent Income / After Tax Permanent Income
  This adjustment is necessary; we need to rescale WIRATIO properly, since WIRATIO obtained 
 using STATA is the ratio of wealth to before tax permanent income, not to after tax permanent income.
 (Note that the work in the MICRODSOP lecture notes takes parameters from Cagetti (2003)
 which is based on after tax income.)
 */

 /*  Income and IncomeRatio are calculated using data in Cagetti (2003) and SCF data  */
 matrix input RawMat = (1.1758, 39497 \ 1.221, 49262 \ 1.2874, 61057 \ 1.2594, 68224 \ 1.4432, 86353 ///
                     \ 1.5055, 96983 \ 1.5509, 98786 \ 1.5663, 1.0223e+005 \ 1.5663, 1e+010 ) 
svmat RawMat
rename RawMat1 RAWIRATIO
rename RawMat2 RAWI	

gen TXIRATIO =.    /* := Before Tax Permanent Income / After Tax Permanent Income */         
local N=_N

qui forvalues i=1/`N' {                                  /* Gives the Before Tax Permanent Income / After Tax Permanent Income RATIO */
 		   replace RAWI = AVGINC[`i'] in 10
           ipolate RAWIRATIO RAWI, gen(TEMP) epolate
		   replace TXIRATIO = TEMP[10] in `i'
		   drop TEMP 
		  }
replace RAWI =. in 10
replace TXIRATIO = 1 if TXIRATIO < 1

/* AfterTax adjustment of Wealth/Income Ratio: AVGWIRATIO represents Net Worth/Before Tax Permanent Income for each HH, 
thus multiplying AVGWIRATIO by TXIRATIO (Bef. Tax Inc / After Tax Inc) gives AVGWIRATIO= Net Worth/After Tax Inc, which is desired.  */	

replace AVGWIRATIO = AVGWIRATIO*TXIRATIO	
  	  
**************************************************************************************************		  

qui: sum AGEID, d
local size=r(max)                        /* size is used as an index number in the following loops*/

gen AVGINCBYAGE = .
gen OBSBYAGE = .

qui foreach p in $percentiles {
  gen `p'INCBYAGE = .
   qui forvalues k = 1/`size' {                                   /* Generates the Average Income for each Age Group defined by AGEID */
           sum AVGINC [aweight = AVGWGT] if AGEID==`k', d     
           replace AVGINCBYAGE = r(mean) if AGEID==`k'
           replace `p'INCBYAGE = r(`p') if AGEID==`k'
		   replace OBSBYAGE = r(N) if AGEID==`k'
		  }
		} 
		
gen AVGNETWBYAGE = .

qui foreach p in $percentiles {
  gen `p'NETWBYAGE = .
  qui forvalues k = 1/`size' {                                   /* Generates the Net Worth for each Age Group defined by AGEID */
           sum AVGNETW [aweight = AVGWGT] if AGEID==`k', d 
           replace AVGNETWBYAGE = r(mean) if AGEID==`k'
           replace `p'NETWBYAGE = r(`p') if AGEID==`k'
		  }		  
		 } 
		 
gen AVGWIRATIOBYAGE = .

qui foreach p in $percentiles {
  gen `p'WIRATIOBYAGE = .
  qui forvalues k = 1/`size' {                                   /* Generates the Wealth/Income Ratio for each Age Group defined by AGEID */
           sum AVGWIRATIO [aweight = AVGWGT] if AGEID==`k', d 
           replace AVGWIRATIOBYAGE = r(mean) if AGEID==`k'
           replace `p'WIRATIOBYAGE = r(`p') if AGEID==`k'
		  }
		 }
		 
gen AGERANGE= `""'
qui forvalues k=1/`size' {
           replace AGERANGE = AgeRange`k' if AGEID==`k'        /* Generates string values for each Age Group which are used in graphs for illustrative purposes */
          }

by AGEID: gen OBS=_n
save "$basePath/Data/Constructed/All2004", replace 

*************************************  2007 Survey Summary ******************************************

use $basePath/Data/Constructed/SCF1992_2007_population, clear
keep if YEAR == 2007

xtset HHID
sort HHID
by HHID: gen OBS=_n
xtset HHID OBS

egen AVGWGT = sum(WGT), by(HHID)          /* Generates one Weight for each Household */
egen AVGINC = sum(INCOME*WGT), by(HHID)   /* This line and the following line Generate the Average Income for each Household */
replace AVGINC = AVGINC/AVGWGT

egen AVGNETW = sum(NETW*WGT), by(HHID)     /* This line and the following line Generate the Average Net Worth for each Household */
replace AVGNETW = AVGNETW/AVGWGT

gen AVGWIRATIO = AVGNETW/AVGINC          /* Generates the Average Wealth/Income Ratio for each Household */

************************************** Age Range Selection *****************************

xtsum HHID
keep if OBS==1
drop OBS 

keep if  AGE >= 26 & AGE <= 60       /* Constructs 5 year period age groups: 31-35, ...., 51-55. */
gen AGEID = int((AGE-26)/5)+1
xtset  AGEID
sort AGEID HHID

***************** Before Tax Permanent Income / After Tax Permanent Income RATIO *****************

/* This section gives the ratio: Before Tax Permanent Income / After Tax Permanent Income
  This adjustment is necessary; we need to rescale WIRATIO properly, since WIRATIO obtained 
 using STATA is the ratio of wealth to before tax permanent income, not to after tax permanent income.
 (Note that the work in the MICRODSOP lecture notes takes parameters from Cagetti (2003)
 which is based on after tax income.)
 */

 /*  Income and IncomeRatio are calculated using data in Cagetti (2003) and SCF data  */
 matrix input RawMat = (1.1758, 39497 \ 1.221, 49262 \ 1.2874, 61057 \ 1.2594, 68224 \ 1.4432, 86353 ///
                     \ 1.5055, 96983 \ 1.5509, 98786 \ 1.5663, 1.0223e+005 \ 1.5663, 1e+010 ) 
svmat RawMat
rename RawMat1 RAWIRATIO
rename RawMat2 RAWI	

gen TXIRATIO =.    /* := Before Tax Permanent Income / After Tax Permanent Income */         
local N=_N

qui forvalues i=1/`N' {                                  /* Gives the Before Tax Permanent Income / After Tax Permanent Income RATIO */
 		   replace RAWI = AVGINC[`i'] in 10
           ipolate RAWIRATIO RAWI, gen(TEMP) epolate
		   replace TXIRATIO = TEMP[10] in `i'
		   drop TEMP 
		  }
replace RAWI =. in 10
replace TXIRATIO = 1 if TXIRATIO < 1

/* AfterTax adjustment of Wealth/Income Ratio: AVGWIRATIO represents Net Worth/Before Tax Permanent Income for each HH, 
thus multiplying AVGWIRATIO by TXIRATIO (Bef. Tax Inc / After Tax Inc) gives AVGWIRATIO= Net Worth/After Tax Inc, which is desired.  */	

replace AVGWIRATIO = AVGWIRATIO*TXIRATIO	
  	  
**************************************************************************************************		  

qui: sum AGEID, d
local size=r(max)                        /* size is used as an index number in the following loops*/

gen AVGINCBYAGE = .
gen OBSBYAGE = .

qui foreach p in $percentiles {
  gen `p'INCBYAGE = .
   qui forvalues k = 1/`size' {                                   /* Generates the Average Income for each Age Group defined by AGEID */
           sum AVGINC [aweight = AVGWGT] if AGEID==`k', d     
           replace AVGINCBYAGE = r(mean) if AGEID==`k'
           replace `p'INCBYAGE = r(`p') if AGEID==`k'
		   replace OBSBYAGE = r(N) if AGEID==`k'
		  }
		} 
		
gen AVGNETWBYAGE = .

qui foreach p in $percentiles {
  gen `p'NETWBYAGE = .
  qui forvalues k = 1/`size' {                                   /* Generates the Net Worth for each Age Group defined by AGEID */
           sum AVGNETW [aweight = AVGWGT] if AGEID==`k', d 
           replace AVGNETWBYAGE = r(mean) if AGEID==`k'
           replace `p'NETWBYAGE = r(`p') if AGEID==`k'
		  }		  
		 } 
		 
gen AVGWIRATIOBYAGE = .

qui foreach p in $percentiles {
  gen `p'WIRATIOBYAGE = .
  qui forvalues k = 1/`size' {                                   /* Generates the Wealth/Income Ratio for each Age Group defined by AGEID */
           sum AVGWIRATIO [aweight = AVGWGT] if AGEID==`k', d 
           replace AVGWIRATIOBYAGE = r(mean) if AGEID==`k'
           replace `p'WIRATIOBYAGE = r(`p') if AGEID==`k'
		  }
		 }
		 
gen AGERANGE= `""'
qui forvalues k=1/`size' {
           replace AGERANGE = AgeRange`k' if AGEID==`k'        /* Generates string values for each Age Group which are used in graphs for illustrative purposes */
          }

by AGEID: gen OBS=_n
save "$basePath/Data/Constructed/All2007", replace  

************************************  2001-2007 Population WIRATIO , AGEID and WEIGHT *******************************
cd $basePath/Data/Constructed/

use All2007, clear
append using All2004
append using All2001
append using All1998
append using All1995
append using All1992

keep HHID YEAR AGEID AGERANGE AVGWIRATIO AVGWGT
sort AGEID AVGWIRATIO 

bysort AGEID: gen N=_N
egen SUMAVGWGT = sum(AVGWGT), by(AGEID)
gen WGTPOP = (AVGWGT/SUMAVGWGT)*N  
gen WIRATIOPOP = AVGWIRATIO

order WIRATIOPOP AGEID WGTPOP
keep WIRATIOPOP AGEID WGTPOP

save "./WIRATIO_Population", replace
save"./SCFdata",replace

cd $basePath
outfile using "./Code/Mathematica/StructuralEstimation/SCFdata.txt", replace

**************************************************************************************************

cd $basePath/$stataPath /* When program ends, make sure working directory is the program's directory */

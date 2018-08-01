
**********************************************************************************
* Epidemiology of Consumption; US results;
**********************************************************************************
clear
#delimit;

set more off;
global basePath `1'; 
*global basePath "D:\slacalek\research\stickyconsumptionus\20180208\cAndCwithStickyE-Latest-MNW-slide-update";
global dataPath $basePath\Code\Empirical\data;  
global logPath $basePath\Code\Empirical;
global docPath $basePath\Tables\;
cd $logPath;
capture log close;
log using $logPath\_usConsDynEmp.log, replace;



global startReg "1960Q1";	* start of the sample;
global endReg "2016Q4";		* end of the sample "2004Q3";
global incomeIndex "3";		* what measure of income?  1 disposable, 2 labor 3 wages&salaries+transfers-socInsurance;
global nwLags "4";			* # of lags for Newey-West std errors;
global ivLags "3";			* # of lags in 1st stage regressions;
global ivLags2 =$ivLags+1;	* # MUST be equal to ivLags plus 1!!;
global ivLagsAlt "4";		

* Instrument sets;	
global ivset1 "L(2/$ivLags).diffcons L(2/$ivLags).dincome L2.diff8cons L2.d8income L(2/$ivLags).wyRatio L(2/$ivLags).dfedfunds L(2/$ivLags).ice"; 
global ivset2 "L(3/$ivLags2).diffcons L(3/$ivLags2).dincome  L3.diff8cons L3.d8income L(3/$ivLags2).wyRatio L(3/$ivLags2).dfedfunds L(3/$ivLags2).ice"; 
global ivsetAlt "L(3/$ivLagsAlt).diffcons L(3/$ivLagsAlt).wyRatio L(3/$ivLagsAlt).dincome"; 

* ivsetAlt is an alternative IV set to ivset1;
* global ivset1 "L(2/$ivLags).diffcons L(2/$ivLags).dincome L(2/$ivLags).wyRatio L(2/$ivLags).dfedfunds L(2/$ivLags).ice"; * L(2/$ivLags).dfedfunds L(2/$ivLags).ice L(2/$ivLags).creditSpread L(2/$ivLags).bigTheta ;
* global ivset2 "L(3/$ivLags2).diffcons L(3/$ivLags2).dincome L(3/$ivLags2).wyRatio L(3/$ivLags2).dfedfunds L(3/$ivLags2).ice"; * L(3/$ivLags2).dfedfunds L(3/$ivLags2).ice L(3/$ivLags2).creditSpread L(3/$ivLags2).bigTheta ;
* global ivsetAlt "L(3/$ivLagsAlt).diffcons L(3/$ivLagsAlt).wyRatio L(3/$ivLagsAlt).dincome"; *L(3/$ivLagsAlt).bigTheta;

* global ivset1 "L(2/$ivLags).diffcons L(2/3).wyRatio L(2/$ivLags).dincome L(2/$ivLags).bigTheta L(2/$ivLags).dfedfunds L(2/$ivLags).ics"; 
* global ivset2 "L(3/$ivLags2).diffcons L(3/4).wyRatio L(3/$ivLags2).dincome L(3/$ivLags2).bigTheta L(3/$ivLags2).dfedfunds L(3/$ivLags2).ics"; 

log off;
import excel using "$dataPath\usData_20170927.xlsx", firstrow sheet("Stata"); 
gen t =  q(1947q1)+_n-1;
format t %tq;
tsset t;

drop if t<q(1959q1);
log on;
do transformData;

* Estimation;
**********************************************************************************;

* NDS consumption;
global consSeries "NDS";
gen diffcons=dndscons;
gen diffcons_ave= dndscons_ave;
gen diff8cons=d8ndscons;
do usConsDynFunc;
mat t1nds=t1;
drop diffcons;
drop diffcons_ave;
drop diff8cons;

tempname hh;
file open `hh' using "$docPath\CampManVsStickyEinner.tex", write replace; file close `hh';
do writeTableResults;

* Nondurable consumption;
global consSeries "Nondurables";
gen diffcons=dndcons;
gen diffcons_ave=dndcons_ave;
gen diff8cons=d8ndcons;
do usConsDynFunc;
mat t1nd=t1;
drop diffcons;
drop diffcons_ave;

file open `hh' using "$docPath\CampManVsStickyEinner.tex", write append;
file write `hh' " \midrule " _n;
file write `hh' " \multicolumn{6}{l}{ Nondurables } \\" _n; 
file write `hh' " \multicolumn{1}{c}{$\Delta \log \mathbf{C}_{t} $} & \multicolumn{1}{c}{$\Delta \log \mathbf{Y}_{t+1}$}& \multicolumn{1}{c}{$ A_{t}  $} & & & \\" _n; file close `hh';

do writeTableResults;

file open `hh' using "$docPath\CampManVsStickyEinner.tex", write append;
file write `hh' " \bottomrule " _n;

file write `hh' "  \multicolumn{6}{p{0.8\textwidth}}{\footnotesize  ";

*file write `hh' " Instruments: \texttt{";
*foreach c of global ivset1 {;
*	file write `hh' "`c' ";
*	};
* file write `hh' "}"; 
file write `hh' "} \\";


* file write `hh' "  \multicolumn{6}{p{0.8\textwidth}}{\footnotesize  ";
* file write `hh' " Time frame: $startReg--$endReg  ";
* file write `hh' "} \\";
* file close `hh';


log on;
mat li t1nds, format(%5.3f);
mat li t1nd, format(%5.3f);

gen dlincomeVariable=D.lincomeVariable;
summ dlincomeVariable if tin($startReg,$endReg), det;
sca varDyD=r(Var);

corr dlincomeVariable L.dlincomeVariable if tin($startReg,$endReg), c;
sca ac1DyD= r(cov_12);

**********************************************************************************;
log close;	

exit, STATA clear;

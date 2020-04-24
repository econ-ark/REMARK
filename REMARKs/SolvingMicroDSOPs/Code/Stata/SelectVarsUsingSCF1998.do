* This file selects variables, using SCF98
* This file closely follows codes written for SAS which create summary variables. 
* (These codes are available at http://www.federalreserve.gov/pubs/oss/oss2/bulletin.macro.txt)  

clear 

** Set memeory 
set memory 32m 

global startDir "`c(pwd)'"
global scfFldr 1998
global scfFile scf98
global SuffixForConstructedFile "_population"

cd ../../../Downloads/SCF/$scfFldr

cap confirm file $scfFile.dta
if _rc~=0 {
  display "File $scfFile is not in the Downloads/SCF/$scfFldr folder; please see ReadMe.txt for instructions."
  exit
}

                   
** Load data and pick up necessary vars from original data
use y1 x42000 x42001 x5729 x7362 x5751 x7650 ///
    x3506 x3507 x3510 x3511 x3514 x3515 x3518 x3519 x3522 x3523 x3526 x3527 x3529 ///
    x3804 x3807 x3810 x3813 x3816 x3818 ///
    x3506 x3507 x9113 x3510 x3511 x9114 x3514 x3515 x9115 x3518 x3519 x9116 x3522 x3523 x9117 x3526 x3527 x9118 x3529 x3706 x9131 x3711 x9132 x3716 x9133 x3718 ///
    x3930 ///
    x3721 x3821 x3822 x3823 x3824 x3825 x3826 x3827 x3828 x3829 x3830 x3915 x3910 x3906 x3908 x7634 x7633 ///
    x3610 x3620 x3630 x3631 ///
    x4216 x4316 x4416 x4816 x4916 x5016 x4226 x4326 x4426 x4826 x4926 x5026 x4227 x4327 x4427 x4827 x4927 x5027 x4231 x4331 x4431 x4831 x4931 x5031 x4234 x4334 x4434 x4834 x4934 x5034 x4436 x5036 ///
    x3902 x4006 x6820 x6826 x6835 x6841 x4018 x4022 x4020 x4026 x4024 x4030 x4028 ///
    x8166 x8167 x8168 x8188 x2422 x2506 x2606 x2623 ///
    x507 x604 x614 x623 x716 x513 x526 x7134 x716 x701 x7133 ///
    x1405 x1409 x1505 x1509 x1605 x1609 x1619 x1703 x1706 x1705 x1803 x1806 x1805 x1903 x1906 x1905 x2002 x2012 x1715 x1815 x1915 x2016 x2723 x2710 x2740 x2727 x2823 x2810 x2840 x2827 x2923 x2910 x2940 x2927 ///
    x3129 x3124 x3126 x3127  x3121 x3122 x3122 x3229 x3224 x3226 x3227  x3221 x3222 x3222 x3329 x3324 x3326 x3327  x3321 x3322 x3322 x3335 x507 x513 x526 x3408 x3412 x3416 x3420 x3424 x3428 ///   
    x4022 x4026 x4030 ///
    x805 x905 x1005 x1108 x1103 x1119 x1114 x1130 x1125 x1136 ///
    x1417 x1517 x1617 x1621 x2006 ///
    x1108 x1103 x1119 x1114 x1130 x1125 x1136 ///
    x427 x413 x421 x430 x424 x7575 ///
    x2218 x2318 x2418 x7169 x2424 x2519 x2619 x2625 x7183 x7824 x7847 x7870 x7924 x7947 x7970 x7179 x1044 x1215 x1219 ///
    x4229 x4230 x4329 x4330 x4429 x4430 x4829 x4830 x4929 x4930 x5029 x5030 ///
    x4010 x3932 x4032 ///
    x14 x19 x8020 x8023 x5902 x5904 x6102 x6104 ///
    using "scf98.dta"

** Generate variables 
* ID
gen ID   = y1                  /* ID # */
gen HHID = (y1-mod(y1,10))/10  /* HH ID # */
gen YEAR = 1998                /* Indicates data is from which wave */

* Weight
gen WGT       = x42001

* Income
gen INCOME     = x5729        /* Income before tax */
replace INCOME = x7362 if x7650!=3
scalar CPIBASE = 2116                   /* September 1992 consumer price index level, the numbers can be found in http://www.federalreserve.gov/pubs/oss/oss2/bulletin.macro.txt*/
scalar CPIADJ = CPIBASE/2405           /* Adjust with CPI (adjusted to 1992$ price) */
scalar CPILAG = 2397/2364               /* Income is the previous year's income level, CPILAG adjust income to survey year */
replace INCOME = INCOME*CPILAG*CPIADJ   /* Adjust with CPI (adjusted to 1992$ price) */   

* Asset 
gen CHECKING = max(0,x3506)*(x3507==5)+max(0,x3510)*(x3511==5) ///
              +max(0,x3514)*(x3515==5)+max(0,x3518)*(x3519==5) ///
              +max(0,x3522)*(x3523==5)+max(0,x3526)*(x3527==5) ///
              +max(0,x3529)*(x3527==5) 
gen SAVING   = max(0,x3804)+max(0,x3807)+max(0,x3810)+max(0,x3813) ///
              +max(0,x3816)+max(0,x3818) 
 gen MMDA    = max(0,x3506)*((x3507==1)*(11<=x9113 & x9113<=13)) ///
              +max(0,x3510)*((x3511==1)*(11<=x9114 & x9114<=13)) ///
              +max(0,x3514)*((x3515==1)*(11<=x9115 & x9115<=13)) ///
              +max(0,x3518)*((x3519==1)*(11<=x9116 & x9116<=13)) ///
              +max(0,x3522)*((x3523==1)*(11<=x9117 & x9117<=13)) ///
              +max(0,x3526)*((x3527==1)*(11<=x9118 & x9118<=13)) ///
              +max(0,x3529)*((x3527==1)*(11<=x9118 & x9118<=13)) ///
              +max(0,x3706)*(11<=x9131 & x9131<=13) ///
              +max(0,x3711)*(11<=x9132 & x9132<=13) ///
              +max(0,x3716)*(11<=x9133 & x9133<=13) ///
              +max(0,x3718)*(11<=x9133 & x9133<=13) 
 gen MMMF    = max(0,x3506)*(x3507==1)*(x9113<11|x9113>13) ///
              +max(0,x3510)*(x3511==1)*(x9114<11|x9114>13) ///
              +max(0,x3514)*(x3515==1)*(x9115<11|x9115>13) ///
              +max(0,x3518)*(x3519==1)*(x9116<11|x9116>13) ///
              +max(0,x3522)*(x3523==1)*(x9117<11|x9117>13) ///
              +max(0,x3526)*(x3527==1)*(x9118<11|x9118>13) ///
              +max(0,x3529)*(x3527==1)*(x9118<11|x9118>13) ///
              +max(0,x3706)*(x9131<11|x9131>13) ///
              +max(0,x3711)*(x9132<11|x9132>13) ///
              +max(0,x3716)*(x9133<11|x9133>13) ///
              +max(0,x3718)*(x9133<11|x9133>13) 
gen MMA      = MMDA+MMMF 
gen CALL     = max(0,x3930) 
gen LIQ      = CHECKING+SAVING+MMA+CALL 

gen CDS      = max(0,x3721) 
 gen STMUTF  = (x3821==1)*max(0,x3822) 
 gen TFBMUTF = (x3823==1)*max(0,x3824) 
 gen GBMUTF  = (x3825==1)*max(0,x3826) 
 gen OBMUTF  = (x3827==1)*max(0,x3828) 
 gen COMUTF  = (x3829==1)*max(0,x3830) 
 gen SNMMF   = TFBMUTF+GBMUTF+OBMUTF+(.5*(COMUTF)) 
 gen RNMMF   = STMUTF + (.5*(COMUTF)) 
gen NMMF     = SNMMF + RNMMF 
gen STOCKS   = max(0,x3915) 
 gen NOTXBND = x3910 
 gen MORTBND = x3906 
 gen GOVTBND = x3908 
 gen OBND    = x7634+x7633 
gen BOND     = NOTXBND + MORTBND + GOVTBND + OBND
 gen IRAKH   = max(0,x3610)+max(0,x3620)+max(0,x3630)
 gen THRIFT  = max(0,x4226)*(x4216==1|x4216==2|x4227==1|x4231==1) ///
              +max(0,x4326)*(x4316==1|x4316==2|x4327==1|x4331==1) ///
              +max(0,x4426)*(x4416==1|x4416==2|x4427==1|x4431==1) ///
              +max(0,x4826)*(x4816==1|x4816==2|x4827==1|x4831==1) ///
              +max(0,x4926)*(x4916==1|x4916==2|x4927==1|x4931==1) ///
              +max(0,x5026)*(x5016==1|x5016==2|x5027==1|x5031==1)
 gen PMOP     = x4436
 replace PMOP = 0 if x4436<=0
 replace PMOP = 0 if x4216!=0 & x4316!=0 & x4416!=0 & x4231!=0 & x4331!=0 & x4431!=0
 replace THRIFT = THRIFT + PMOP 
 replace PMOP   = x5036
 replace PMOP = 0 if x5036<=0
 replace PMOP = 0 if x4816!=0 & x4916!=0 & x5016!=0 & x4831!=0 & x4931!=0 & x5031!=0
 replace THRIFT = THRIFT + PMOP
gen RETQLIQ  = IRAKH + THRIFT
gen SAVBND   = x3902
gen CASHLI   = max(0,x4006)
 gen RANNUIT  = 0
 gen SANNUIT  = 0
 gen CANNUIT  = 0
 gen RTRUST   = 0
 gen STRUST   = 0
 gen CTRUST   = 0
 replace RANNUIT = x6820 if x6826== 1|x6826==3
 replace SANNUIT = x6820 if x6826== 2|x6826==7
 replace CANNUIT = x6820 if x6826== 5|x6826==6|x6826==8|x6826==9|x6826==-7
 replace RTRUST  = x6835 if x6841==1|x6841==3
 replace STRUST  = x6835 if x6841==2|x6841==7
 replace CTRUST  = x6835 if x6841==5|x6841==6|x6841==8|x6841==9|x6841==-7
 gen ROTHMA   = max(0,(RANNUIT + RTRUST))
 gen SOTHMA   = max(0,(SANNUIT + STRUST))
 gen COTHMA   = max(0,(CANNUIT + CTRUST))
gen OTHMA    = ROTHMA+SOTHMA+COTHMA 
gen OTHFIN   = x4018+x4022*(x4020==62|x4020==63|x4020==64|x4020==66|x4020==71|x4020==73|x4020==74|x4020==-7) ///
              +x4026*(x4024==62|x4024==63|x4024==64|x4024==66|x4024==71|x4024==73|x4024==74|x4024==-7) ///
              +x4030*(x4028==62|x4028==63|x4028==64|x4028==66|x4028==71|x4028==73|x4028==74|x4028==-7)  
gen FIN      = LIQ+CDS+NMMF+STOCKS+BOND+RETQLIQ+SAVBND+CASHLI+OTHMA+OTHFIN /* Total fin asset */

gen VEHIC    = max(0,x8166)+max(0,x8167)+max(0,x8168)+max(0,x8188) ///
              +max(0,x2422)+max(0,x2506)+max(0,x2606)+max(0,x2623)
replace x507 = 9000 if x507 > 9000
gen HOUSES   = (x604+x614+x623+x716) + ((10000-max(0,x507))/10000)*(x513+x526)
 * replace HOUSES = (x7134/10000)*x716 if x701==-7 & x7133==1
gen ORESRE   = max(x1405,x1409)+max(x1505,x1509)+max(x1605,x1609)+max(0,x1619) ///
              +(x1703==12|x1703==14|x1703==21|x1703==22|x1703==25|x1703==40|x1703==41|x1703==42|x1703==43|x1703==44|x1703==49|x1703==50|x1703==52|x1703==999) ///
              *max(0,x1706)*(x1705/10000) ///
              +(x1803==12|x1803==14|x1803==21|x1803==22|x1803==25|x1803==40|x1803==41|x1803==42|x1803==43|x1803==44|x1803==49|x1803==50|x1803==52|x1803==999) ///
              *max(0,x1806)*(x1805/10000) ///
              +(x1903==12|x1903==14|x1903==21|x1903==22|x1903==25|x1903==40|x1903==41|x1903==42|x1903==43|x1903==44|x1903==49|x1903==50|x1903==52|x1903==999) ///
              *max(0,x1906)*(x1905/10000) ///
              +max(0,x2002)
gen NNRESRE  = (x1703==1|x1703==2|x1703==3|x1703==4|x1703==5|x1703==6|x1703==7|x1703==10|x1703==11|x1703==13|x1703==15|x1703==24|x1703==45|x1703==46|x1703==47|x1703==48|x1703==51|x1703==-7) ///
              *(max(0,x1706)*(x1705/10000)-x1715*(x1705/10000)) ///
              +(x1803==1|x1803==2|x1803==3|x1803==4|x1803==5|x1803==6|x1803==7|x1803==10|x1803==11|x1803==13|x1803==15|x1803==24|x1803==45|x1803==46|x1803==47|x1803==48|x1803==51|x1803==-7) ///
              *(max(0,x1806)*(x1805/10000)-x1815*(x1805/10000)) ///
              +(x1903==1|x1903==2|x1903==3|x1903==4|x1903==5|x1903==6|x1903==7|x1903==10|x1903==11|x1903==13|x1903==15|x1903==24|x1903==45|x1903==46|x1903==47|x1903==48|x1903==51|x1903==-7) ///
              *(max(0,x1906)*(x1905/10000)-x1915*(x1905/10000)) ///
              +max(0,x2012)-x2016
replace NNRESRE = NNRESRE-x2723*(x2710==78)-x2740*(x2727==78)-x2823*(x2810==78) ///
                 -x2840*(x2827==78)-x2923*(x2910==78)-x2940*(x2927==78) if NNRESRE!=0
gen FLAG781  = (NNRESRE!=0)
gen BUS      = max(0,x3129)+max(0,x3124)-max(0,x3126)*(x3127==5) ///
              +max(0,x3121)*(x3122==1|x3122==6) ///
              +max(0,x3229)+max(0,x3224)-max(0,x3226)*(x3227==5) ///
              +max(0,x3221)*(x3222==1|x3222==6) ///
              +max(0,x3329)+max(0,x3324)-max(0,x3326)*(x3327==5) ///
              +max(0,x3321)*(x3322==1|x3322==6) ///
              +max(0,x3335)+(x507/10000)*(x513+x526) ///
              +max(0,x3408)+max(0,x3412)+max(0,x3416)+max(0,x3420) ///
              +max(0,x3424)+max(0,x3428)
gen OTHNFIN  = x4022 + x4026 + x4030 - OTHFIN + x4018
gen NFIN     = VEHIC+HOUSES+ORESRE+NNRESRE+BUS+OTHNFIN
gen ASSET    = FIN+NFIN /* Total asset */

* Debt 
gen MRTHEL   = x805+x905+x1005+x1108*(x1103==1)+x1119*(x1114==1) ///
              +x1130*(x1125==1)+max(0,x1136)*(x1108*(x1103==1)+x1119*(x1114==1) ///
              +x1130*(x1125==1))/(x1108+x1119+x1130) if (x1108+x1119+x1130)>=1
replace MRTHEL = x805+x905+x1005+.5*(max(0,x1136)) if (x1108+x1119+x1130)<1
 gen MORT1    = (x1703==12|x1703==14|x1703==21|x1703==22|x1703==25|x1703==40|x1703==41|x1703==42|x1703==43|x1703==44|x1703==49|x1703==50|x1703==52|x1703==999) ///
               *x1715*(x1705/10000)
 gen MORT2    = (x1803==12|x1803==14|x1803==21|x1803==22|x1803==25|x1803==40|x1803==41|x1803==42|x1803==43|x1803==44|x1803==49|x1803==50|x1803==52|x1803==999) ///
               *x1815*(x1805/10000)
 gen MORT3    = (x1903==12|x1903==14|x1903==21|x1903==22|x1903==25|x1903==40|x1903==41|x1903==42|x1903==43|x1903==44|x1903==49|x1903==50|x1903==52|x1903==999) ///
               *x1915*(x1905/10000)
gen RESDBT   = x1417+x1517+x1617+x1621+MORT1+MORT2+MORT3+x2006
 gen FLAG782  = (FLAG781!=1 & ORESRE>0)
replace RESDBT = RESDBT+x2723*(x2710==78)+x2740*(x2727==78)+x2823*(x2810==78)+x2840*(x2827==78) ///
                +x2923*(x2910==78)+x2940*(x2927==78) if FLAG781!=1 & ORESRE>0
 gen FLAG67   = (ORESRE>0)
replace RESDBT= RESDBT+x2723*(x2710==67)+x2740*(x2727==67)+x2823*(x2810==67)+x2840*(x2827==67) ///
               +x2923*(x2910==67)+x2940*(x2927==67) if ORESRE>0 
gen OTHLOC   = x1108*(x1103!=1)+x1119*(x1114!=1)+x1130*(x1125!=1) ///
              +max(0,x1136)*(x1108*(x1103!=1)+x1119*(x1114!=1) ///
              +x1130*(x1125!=1))/(x1108+x1119+x1130) if (x1108+x1119+x1130)>=1
replace OTHLOC = .5*(max(0,x1136)) if (x1108+x1119+x1130)<1 
gen CCBAL    = max(0,x427)+max(0,x413)+max(0,x421)+max(0,x430)+max(0,x424)+max(0,x7575)
gen INSTALL  = x2218+x2318+x2418+x7169+x2424+x2519+x2619+x2625+x7183 ///
              +x7824+x7847+x7870+x7924+x7947+x7970+x7179+x1044+x1215+x1219
replace INSTALL = INSTALL+x2723*(x2710==78)+x2740*(x2727==78) ///
                 +x2823*(x2810==78)+x2840*(x2827==78)+x2923*(x2910==78)+x2940*(x2927==78) ///
                  if FLAG781==0 & FLAG782==0
replace INSTALL = INSTALL+x2723*(x2710==67)+x2740*(x2727==67)+x2823*(x2810==67) ///
                 +x2840*(x2827==67)+x2923*(x2910==67)+x2940*(x2927==67) if FLAG67==0
replace INSTALL = INSTALL+x2723*(x2710!=67&x2710!=78)+x2740*(x2727!=67&x2727!=78) ///
                 +x2823*(x2810!=67&x2810!=78)+x2840*(x2827!=67&x2827!=78)+x2923*(x2910!=67&x2910!=78) ///
                 +x2940*(x2927!=67&x2927!=78)
gen PENDBT   = max(0,x4229)*(x4230==5)+max(0,x4329)*(x4330==5) ///
              +max(0,x4429)*(x4430==5)+max(0,x4829)*(x4830==5) ///
              +max(0,x4929)*(x4930==5)+max(0,x5029)*(x5030==5)
gen CASHLIDB = max(0,x4010)
gen CALLDBT  = max(0,x3932)
gen ODEBT    = max(0,x4032)
gen DEBT     = MRTHEL+RESDBT+OTHLOC+CCBAL+INSTALL+PENDBT+CASHLIDB+CALLDBT+ODEBT /* Total debt */

* Net worth 
gen NETW     = ASSET-DEBT
replace NETW = NETW*CPIADJ

* Ratio of net worth to income 
gen WIRATIO  = NETW/INCOME    


* Demographic vars
gen AGE     = x14       /* Age */
gen MARITST = x8023     /* Marital status */
keep if MARITST == 1    /* Keep if married following Cagetti(2003) */

gen EDUC     = 0
replace EDUC = 1 if x5902!=1 /* No high school deg */
replace EDUC = 2 if x5902==1 /* High school deg */
replace EDUC = 3 if x5904==1 /* College deg */
* keep if EDUC == 3            /* Keep college graduates only */

* Correct time effects. The base is set at 25 yrs old in 1980 (0 yrs old in 1955) 
replace INCOME = INCOME/exp(0.016*(YEAR-1955-AGE))
drop if INCOME<=0
replace NETW   = NETW/exp(0.016*(YEAR-1955-AGE))


** Keep necessary vars
keep HHID YEAR WGT INCOME NETW WIRATIO AGE

** Save data
cd "$startDir"
cd ../../Data/Constructed
** Save data
save "./SCF$scfFldr$SuffixForConstructedFile", replace 

** End in the same directory you started from
cd "$startDir"

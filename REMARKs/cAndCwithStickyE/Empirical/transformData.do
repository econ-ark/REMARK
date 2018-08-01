
#delimit;
label var t "Time";

gen pcdefl=(ipdefcon*pop)/100;
gen wealth=(networth/pcdefl);

gen tcons=(conexp/pcdefl);
gen dcons=(dconexp/pcdefl);
gen ndcons=(nconexp/pcdefl);
gen scons=(sconexp/pcdefl);
gen ndscons=(nconexp+sconexp)/pcdefl;
gen ydisp=(disposy/pcdefl);
gen lydisp = log(ydisp);
gen dydisp = D.lydisp;
gen ytransf=(transfy/pcdefl);
gen ypty=((propiny+rentaly+dividey+interey)/pcdefl);
gen ndstotalconsrat=ndscons/tcons;

gen lincome_co=wagesal+transfy-socins;
gen taxes=perstax*wagesal/(wagesal+propiny+rentaly+dividey+interey);
gen lincome_ll=wagesal+transfy+wagesalsuppl-socins-taxes;

gen lndscons=log(ndscons);
gen lndcons=log(ndcons);
gen ltcons=log(tcons);
gen lnds=log(ndscons);
gen lwagesal=log(wagesal);
gen lincome_co_real=lincome_co/pcdefl;
gen llincome_co=log(lincome_co/pcdefl);
gen dllincome_co=D.llincome_co;
gen lincome_ll_real=lincome_ll/pcdefl;
gen llincome_ll=log(lincome_ll/pcdefl);
gen dllincome_ll=D.llincome_ll;

label var llincome_co "Labor Income a la CFW (1994)";
label var llincome_ll "Labor Income a la LL (2004)";
label var lydisp "Disposable Income";

gen dndscons=D.lndscons;
gen dndcons=D.lndcons;
gen dtcons=D.ltcons;

gen d8ndscons=lndscons-L8.lndscons;
gen d8ndcons=lndcons-L8.lndcons;
gen d8tcons=ltcons-L8.ltcons;

gen dndscons_ave=(dndscons+L.dndscons+L2.dndscons+L3.dndscons)/4; 
gen dndcons_ave=(dndcons+L.dndcons+L2.dndcons+L3.dndcons)/4; 
gen dtcons_ave=(dtcons+L.dtcons+L2.dtcons+L3.dtcons)/4; 

gen muExp=icsq_12;
gen un=unp;
gen dfedfunds=D.fedfunds;

gen lincomeVariable=lydisp;
gen incomeVariable=.;

if $incomeIndex==2 {;
	replace lincomeVariable=llincome_ll;
	replace incomeVariable=lincome_ll_real;
	};
else if $incomeIndex==3 {;
	replace lincomeVariable=llincome_co;
	replace incomeVariable=lincome_co_real;
	};

gen dincome = D.lincomeVariable;
gen d8income=lincomeVariable-L8.lincomeVariable;
gen dincome_ave = (dincome+L.dincome+L2.dincome+L3.dincome)/4;
gen linTrend=_n;
reg lincomeVariable linTrend if tin($startReg,$endReg);
sca constant=_b[linTrend];

gen wyRatio=wealth/incomeVariable;



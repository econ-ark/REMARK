
*******************************
** Privileged & Confidential **
*******************************
clear
capture log close
set more off
set type double

global scripts = 	c(pwd)
global scripts = 	subinstr("$scripts","\","/",.)
global path 		"$scripts/.."

***
forval year=1/80{

	if `year' < 10 import delimited using "$path/year0`year'.txt", clear
	if `year' >= 10 import delimited using "$path/year`year'.txt", clear

	di "Importing `year'"
	assert _N == 1203

	* Match observations 
	gen obs = _n 
	replace obs = obs - (1203/3) if obs > (1203/3)
	replace obs = obs - (1203/3) if obs > (1203/3)

	* Seperate variables
	gen var = ""
	replace var = "RiskyShare" if _n <= (1203/3)
	replace var = "Consumption" if _n > (1203/3) & _n <= (1203/3)*2
	replace var = "Value" if mi(var)

	* Reshape
	reshape wide v1, i(obs) j(var) str
	ds obs, not
	foreach var in `r(varlist)'{
		local newname = subinstr("`var'","v1","",.)
		ren `var' `newname'
	}

	* Agent cash
	ren obs Cash
	replace Cash = Cash - 1

	* Agents age
	gen Age = `year' + 20

	save "$scripts/`year'.dta", replace

} 


* Combine
clear
forval year=1/80{
	append using "$scripts/`year'.dta"
}

* Clean up
forval year=1/80{
	erase "$scripts/`year'.dta"
}

* Output
order Age Cash RiskyShare Consumption Value
export excel using "$path/Combined Fortran data.xlsx", sheet("r.Data") firstrow(variables) sheetreplace


* EOF
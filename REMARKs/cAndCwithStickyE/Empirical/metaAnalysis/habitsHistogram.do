clear
clear mata
set more off

global basePath `1'
cd "$basePath\Code\Empirical\metaAnalysis"
use "habit.dta", clear

*set scheme ecb2015


************************************
* VARIABLES DEFINITIONS AND LABELS *
************************************

drop if missing(se)

replace avg_year               = avg_year - 1932
replace no_year                = ln(no_year)
replace no_cross               = ln(no_cross)
replace citations              = ln(1+citations)/(2015-pub_year) 
replace pub_year               = pub_year-1991
gen invsqrtn 				   = 1/(sqrt(observations))
replace observations           = ln(observations)
bysort idstudy: egen habit_med = median(habit)

* Number of estimates in a given study
bysort idstudy: egen no_est = max(id)
gen inv_no_est              = 1/no_est
********************************************

global drop "1" 
* dropping estimates of habits smaller than -1 or larger than 1.5 (following the EER paper)

if $drop == 1 {

drop if habit < -1 | habit > 1.5

} 

* Fig: Histogram all estimates
sum habit, detail
global m_all = r(p50)

* Fig: Histogram micro and macro separate
sum habit if micro == 1, detail
global m_micro = r(p50)
sum habit if micro == 0, detail
global m_macro = r(p50)

* Fig: Histogram micro and macro separate non-overlapping                        NOTE: medians in this type of graph need to be inserted manually ex-post in the editor. For the values, check $m_micro and $m_macro                    
sum habit if micro == 1, detail
global m_micro = r(p50)
sum habit if micro == 0, detail
global m_macro = r(p50)
gen habit_b = round(habit, 0.1)
preserve
collapse (count) counta=habit, by(habit_b micro)

// From here fill in gaps in  categories
local total = _N 
set obs `=_N+10'
replace micro = 0 if _n > `total'
replace counta = 0 if _n > `total'
replace habit_b = -2.1 + 0.1*(_n-`total') if _n > `total'

local total = _N
set obs `=_N+4'
replace micro = 0 if _n > `total'
replace counta = 0 if _n > `total'
replace habit_b = -1 + 0.1*(_n-`total') if _n > `total'

local total = _N
set obs `=_N+8'
replace micro = 0 if _n > `total'
replace counta = 0 if _n > `total'
replace habit_b = 1.2 if _n == `total'+1
replace habit_b = 1.4 if _n == `total'+2
replace habit_b = 1.6 if _n == `total'+3
replace habit_b = 1.7 if _n == `total'+4
replace habit_b = 1.9 if _n == `total'+5
replace habit_b = 2.1 if _n == `total'+6
replace habit_b = 2.2 if _n == `total'+7
replace habit_b = 2.3 if _n == `total'+8
                                                                
if $drop == 1 {

drop if habit_b < -1 | habit_b > 1.5

graph bar counta, over(micro) over(habit_b, relabel(1 "-1" 2 " " 3 " " 4 " " 5 " " 6 "-0.5" 7 " " 8 " " 9 " " 10 " " 11 "0" 12 " " 13 " " 14 " " 15 " " 16 "0.5" 17 " " 18 " " 19 " " 20 " " 21 "1" 22 " " 23 " " 24 " " 25 " " 26 "1.5")) /// 
legend(order(1 "Macro" 2 "Micro")) asyvars ytitle("Frequency (Number of estimates in studies)") b1title("Habit persistence {&chi}") graphregion(color(white)) bar(1, color(blue*2)) bar(2, color(blue*0.6))
graph export "$basePath\Figures\microMacroMetaHistogram.png", width(2000) replace
graph export "$basePath\Figures\microMacroMetaHistogram.pdf", replace
graph export "$basePath\Figures\microMacroMetaHistogram.svg", replace

} 

else {

graph bar counta, over(micro) over(habit_b, relabel(1 " " 2 "-2" 3 " " 4 " " 5 " " 6 " " 7 "-1.5" 8 " " 9 " " 10 " " 11 " " 12 "-1" 13 " " 14 " " 15 " " 16 " " 17 "-0.5" 18 " " 19 " " 20 " " 21 " " 22 "0" 23 " " 24 " " 25 " " 26 " " 27 "0.5" 28 " " 29 " " 30 " " 31 " " 32 "1" 33 " " 34 " " 35 " " 36 " " 37 "1.5" 38 " " 39 " " 40 " " 41 " " 42 "2" 43 " " 44 " " 45 " " 46 " ")) /// 
legend(order(1 "Macro" 2 "Micro" )) asyvars ytitle("Frequency (Number of estimates in studies)") xtitle("Habit persistence {&chi}")
graph export "$basePath\Figures\microMacroMetaHistogram.png", width(2000) replace 
graph export "$basePath\Figures\microMacroMetaHistogram.pdf", replace 
graph export "$basePath\Figures\microMacroMetaHistogram.svg", replace 

}

restore 

exit, STATA clear

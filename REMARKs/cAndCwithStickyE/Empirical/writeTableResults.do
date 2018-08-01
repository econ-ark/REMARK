
#delimit;
tempname hh;
file open `hh' using "$docPath\CampManVsStickyEinner.tex", write append;
file write `hh' _n;

sca ptest = 2*(1-normal(abs(m1a1)/m1se1));
do MakeTestStringJ;
file write `hh'  %5.3f (m1a1) (teststr) "  & & & OLS & " %5.3f (m1r2) " & \\"_n;
file write `hh' " (" %5.3f (m1se1) ")  &  &  & &  &  \\"_n;

sca ptest = 2*(1-normal(abs(m4a1)/m4se1));
do MakeTestStringJ;
file write `hh'  %5.3f (m4a1) (teststr) "  & & & IV & " %5.3f (m4r2) " & " %5.3f (m4p1) " \\"_n;
file write `hh' "   (" %5.3f (m4se1) ")  &  &  & &  & "  %5.3f (m4p2)  " \\"_n;

sca ptest = 2*(1-normal(abs(m2a2)/m2se2));
do MakeTestStringJ;
file write `hh' "    & " %5.3f (m2a2) (teststr) " & & IV & " %5.3f (m2r2) " & " %5.3f (m2p1) " \\"_n;
file write `hh' "   &  (" %5.3f (m2se2) ")  &  & &  & " %5.3f (m2p2) "  \\"_n;

sca ptest = 2*(1-normal(abs(m3a3)/m3se3));
sca m3a3 = 10000*m3a3;
sca m3se3 = 10000*m3se3;
do MakeTestStringJ;
file write `hh' "    &  & " %9.2f (m3a3) "\text{e$-4$}" (teststr) "  & IV & " %5.3f (m3r2) " & " %5.3f (m3p1) " \\"_n;
file write `hh' "   &  &  (" %9.2f (m3se3) "\text{e$-4$})  & & & " %5.3f (m3p2)  "  \\"_n;

sca ptest = 2*(1-normal(abs(m7a1)/m7se1));
do MakeTestStringJ;
file write `hh'  %5.3f (m7a1) (teststr) " & ";
sca ptest = 2*(1-normal(abs(m7a2)/m7se2));
do MakeTestStringJ;
file write `hh'  %5.3f (m7a2) (teststr) " & ";
sca ptest = 2*(1-normal(abs(m7a3)/m7se3));
sca m7a3 = 10000*m7a3;
sca m7se3 = 10000*m7se3;
do MakeTestStringJ;
file write `hh'  %9.2f (m7a3) "\text{e$-4$}" (teststr) " & IV & " %5.3f (m7r2) " & " %5.3f (m7p1) " \\"_n;
file write `hh' "  (" %5.3f (m7se1) ") & (" %5.3f (m7se2) ") &  (" %9.2f (m7se3) "\text{e$-4$})  & & & " %5.3f (m7p2)  " \\"_n;

file write `hh' "  \multicolumn{6}{l}{ Memo: For instruments $\mathbf{Z}_t, \Delta\log\mathbf{C}_{t}=\mathbf{Z}_t \zeta, ~\bar{R}^2= ";
file write `hh' %5.3f (m7rC1) " $ } \\"_n;

file close `hh';

**********************************************************;
tempname hh;
file open `hh' using "$docPath\CampManVsStickyEinner_$consSeries.tex", write replace;

sca ptest = 2*(1-normal(abs(m1a1)/m1se1));
do MakeTestStringJ;
file write `hh'  %5.3f (m1a1) (teststr) "  & & & OLS & " %5.3f (m1r2) " & \\"_n;
file write `hh' " (" %5.3f (m1se1) ")  &  &  & &  &  \\"_n;

sca ptest = 2*(1-normal(abs(m4a1)/m4se1));
do MakeTestStringJ;
file write `hh'  %5.3f (m4a1) (teststr) "  & & & IV & " %5.3f (m4r2) " & " %5.3f (m4p1) " \\"_n;
file write `hh' "   (" %5.3f (m4se1) ")  &  &  & &  & "  %5.3f (m4p2)  " \\"_n;

sca ptest = 2*(1-normal(abs(m2a2)/m2se2));
do MakeTestStringJ;
file write `hh' "    & " %5.3f (m2a2) (teststr) " & & IV & " %5.3f (m2r2) " & " %5.3f (m2p1) " \\"_n;
file write `hh' "   &  (" %5.3f (m2se2) ")  &  & &  & " %5.3f (m2p2) "  \\"_n;

sca ptest = 2*(1-normal(abs(m3a3)/m3se3));

do MakeTestStringJ;
file write `hh' "    &  & " %9.2f (m3a3) "\text{e$-4$}" (teststr) "  & IV & " %5.3f (m3r2) " & " %5.3f (m3p1) " \\"_n;
file write `hh' "   &  &  (" %9.2f (m3se3) "\text{e$-4$})  & & & " %5.3f (m3p2)  "  \\"_n;

sca ptest = 2*(1-normal(abs(m7a1)/m7se1));
do MakeTestStringJ;
file write `hh'  %5.3f (m7a1) (teststr) " & ";
sca ptest = 2*(1-normal(abs(m7a2)/m7se2));
do MakeTestStringJ;
file write `hh'  %5.3f (m7a2) (teststr) " & ";
sca ptest = 2*(1-normal(abs(m7a3)/m7se3));
do MakeTestStringJ;
file write `hh'  %9.2f (m7a3) "\text{e$-4$}" (teststr) " & IV & " %5.3f (m7r2) " & " %5.3f (m7p1) " \\"_n;
file write `hh' "  (" %5.3f (m7se1) ") & (" %5.3f (m7se2) ") &  (" %9.2f (m7se3) "\text{e$-4$})  & & & " %5.3f (m7p2)  " \\"_n;

file write `hh' "  \multicolumn{6}{l}{ Memo: For instruments $\mathbf{Z}_t, \Delta\log\mathbf{C}_{t}=\mathbf{Z}_t \zeta, ~\bar{R}^2= ";
file write `hh' %5.3f (m7rC1) " $ } \\"_n;

file close `hh';


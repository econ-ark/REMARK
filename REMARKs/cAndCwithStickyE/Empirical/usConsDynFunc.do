
#delimit;

* Model 1: n=0, lagged consumption only;
reg diffcons L.diffcons if tin($startReg,$endReg);	* OLS;
scalar m1r2 = e(r2_a);
newey  diffcons L.diffcons if tin($startReg,$endReg), lag($nwLags);	* HAC errors;
scalar m1a1 = _b[L.diffcons];
scalar m1se1 = _se[L.diffcons];
scalar m1t1 = m1a1/m1se1;
scalar m1n = e(N);

* Model 2: n=0, predicted (instrumented) income only;
reg dincome $ivset1 if tin($startReg,$endReg);
scalar m2r1 = e(r2_a);
predict dincomeHatm2;
reg diffcons dincomeHatm2 if tin($startReg,$endReg);
scalar m2r2 = e(r2_a);

ivreg2 diffcons (dincome = $ivset1) if tin($startReg,$endReg), robust bw($nwLags) ffirst;	* IV (kitchen sink);
scalar m2a2= _b[dincome];
scalar m2se2=_se[dincome];
scalar m2t2=m2a2/m2se2;
matrix efirst = e(first);
scalar m2p1 = efirst[6,1];	* First stage p value;
scalar m2p1 = e(idp);		* KP p value;
scalar m2f1 = efirst[3,1];	* First stage F stat;
scalar m2p2 = e(jp);		* Hansen's J stat;
scalar m2n = e(N);

* Model 3: n=0, wyRatio only;
reg wyRatio $ivset1 if tin($startReg,$endReg);
scalar m3r1 = e(r2_a);
predict wyRatioHatm3;
reg diffcons wyRatioHatm3 if tin($startReg,$endReg);
scalar m3r2 = e(r2_a);

ivreg2 diffcons (wyRatio = $ivset2) if tin($startReg,$endReg), robust bw($nwLags) ffirst;	* IV (kitchen sink);
scalar m3a3= _b[wyRatio];
scalar m3se3=_se[wyRatio];
scalar m3t3=m3a3/m3se3;
matrix efirst = e(first);
scalar m3p1 = efirst[6,1];	* First stage p value;
scalar m3p1 = e(idp);		* KP p value;
scalar m3f1 = efirst[3,1];	* First stage F stat;
scalar m3p2 = e(jp);		* Hansen's J stat;
scalar m3n = e(N);

*reg diffcons L.wyRatio if tin($startReg,$endReg);	* OLS;
*scalar m3r2 = e(r2_a);
*newey  diffcons L.wyRatio if tin($startReg,$endReg), lag($nwLags);	* HAC errors;
*scalar m3a3 = _b[L.wyRatio];
*scalar m3se3 = _se[L.wyRatio];
*scalar m3t3 = m3a3/m3se3;
*scalar m3n = e(N);


* Model 4: n=1, predicted (instrumented) consumption only;
reg diffcons $ivset1 if tin($startReg,$endReg);
scalar m4r1 = e(r2_a);
predict dconsHatm4;
reg diffcons L.dconsHatm4 if tin($startReg,$endReg);
scalar m4r2 = e(r2_a);

ivreg2 diffcons (L.diffcons = $ivset2) if tin($startReg,$endReg), robust bw($nwLags) ffirst;	* IV (kitchen sink);
scalar m4a1= _b[L.diffcons];
scalar m4se1=_se[L.diffcons];
scalar m4t1 = m4a1/m4se1;
matrix efirst = e(first);
*scalar m4p1 = efirst[6,1];
scalar m4p1 = e(idp);		* KP p value;
scalar m4f1 = efirst[3,1];
scalar m4p2 = e(jp);
scalar m4n = e(N);

ivreg2 diffcons (L.diffcons = $ivsetAlt) if tin($startReg,$endReg), robust bw($nwLags) ffirst;	* IV (kitchen sink);


* Model 7: n=1, predicted (instrumented) consumption, íncome and wyRatio;
reg diffcons $ivset1 if tin($startReg,$endReg);
predict diffconsHatm7;
scalar m7rC1 = e(r2_a);
reg dincome $ivset2 if tin($startReg,$endReg);
predict dincomeHatm7;
scalar m7r1inc = e(r2_a);
reg wyRatio $ivset1 if tin($startReg,$endReg);
predict wyRatioHatm7;
reg diffcons L.diffconsHatm7 dincomeHatm7 L.wyRatioHatm7 if tin($startReg,$endReg);;
scalar m7r2 = e(r2_a);

ivreg2 diffcons (L.diffcons dincome L.wyRatio = $ivset2) if tin($startReg,$endReg), robust bw($nwLags) ffirst;	* IV (kitchen sink);
scalar m7a1= _b[L.diffcons];
scalar m7se1=_se[L.diffcons];
scalar m7t1=m7a1/m7se1;
scalar m7a2= _b[dincome];
scalar m7se2=_se[dincome];
scalar m7t2=m7a2/m7se2;
scalar m7a3= _b[L.wyRatio];
scalar m7se3=_se[L.wyRatio];
scalar m7t3=m7a3/m7se3;
scalar m7p1 = e(idp);		* KP p value;
scalar m7p2 = e(jp);
scalar m7n = e(N);


ivreg2 diffcons (L.diffcons dincome L.wyRatio = $ivsetAlt) if tin($startReg,$endReg), robust bw($nwLags) ffirst;	* IV (kitchen sink);

qui summ diffcons if tin($startReg,$endReg);
scalar varC = r(sd)/100;
scalar bBase = m4a1;


log off;
* put it all together;
**********************************************************************************;
mat t1=(
m1a1,.a,.a,.c,m1r2,.a,m1n\
m1se1,.a,.a,.a,.a,.a,.a\
m1t1,.a,.a,.a,.a,.a,.a\
.a,m2a2,.a,.b,m2r2,m2p1,m2n\
.a,m2se2,.a,.a,m2r1,m2p2,.a\
.a,m2t2,.a,.a,.a,m2f1,.a\
.a,.a,m3a3,.c,m3r2,.a,m3n\
.a,.a,m3se3,.a,.a,.a,.a\
.a,.a,m3t3,.a,.a,.a,.a\
m4a1,.a,.a,.b,m4r2,m4p1,m4n\
m4se1,.a,.a,.a,m4r1,m4p2,.a\
m4t1,.a,.a,.a,.a,m4f1,.a\
m7a1,m7a2,m7a3,.b,m7r2,m7p1,m7n\
m7se1,m7se2,m7se3,.a,.a,m7p2,.a\
m7t1,m7t2,m7t3,.a,.a,.a,.a
);

mat colnames t1 = "a1" "a2" "a3" "olsIv" "r2" "oid" "nobs";
mat rownames t1 = "M1" "-" "-" "M2" "-" "-" "M3" "-" "-" "M4" "-" "-" "M7" "-" "-";

drop dincomeHatm2 wyRatioHatm3 dconsHatm4 diffconsHatm7 dincomeHatm7 wyRatioHatm7;

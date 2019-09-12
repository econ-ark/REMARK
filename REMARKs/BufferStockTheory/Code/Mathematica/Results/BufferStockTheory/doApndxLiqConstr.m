(* ::Package:: *)

(* This cell contains generic setup stuff to prepare for execution of the programs *)
ClearAll["Global`*"]; ParamsAreSet = False;
If[$VersionNumber < 6,(*then*) Print["These programs require Mathematica version 6 or greater."]; Abort[]];
(* If running from Notebook front end, set directory to Notebook's directory *)
If[Length[$FrontEnd] > 0, NBDir = SetDirectory[NotebookDirectory[]]];
(* If not running from Notebook front end, set directory manually *)
If[Length[$FrontEnd] == 0,SetDirectory["/Volumes/Data/Work/BufferStock/BufferStockTheory/Latest/Code/Mathematica/Results/BufferStockTheory"]];
SaveFigs = True;

HomeDir = Directory[];
CodeDir = HomeDir<>"/../../CoreCode";
CDToHomeDir := SetDirectory[HomeDir];
CDToCodeDir := SetDirectory[HomeDir<>"/../../CoreCode"];
CDToCodeDir;
<< SetupModelSolutionRoutines.m;
<< SetParamsToBaselineVals.m;
CDToHomeDir;


(* Example where PF-GIC fails but RIC holds *)
Clear[R,\[Beta],\[CapitalGamma],\[Rho],m\[Sharp],\[ScriptC]\[EmptySmallCircle]];
Print["{R,\[Beta],\[CapitalGamma],\[Rho]}=",{R=1.02,\[Beta]=1/1.02,\[CapitalGamma]=0.98,\[Rho]=2}];
Print["PF-GIC fails (\[CapitalGamma] < \[CapitalThorn]):" ,\[CapitalGamma] < \[CapitalThorn]];
Print["{\[CapitalGamma] , \[CapitalThorn]=(R\!\(\*SuperscriptBox[\()\), \(1/\[Rho]\)]\) \!\(\*SuperscriptBox[\(\[Beta]\), \(1/\[Rho]\)]\) , (R)}=",{\[CapitalGamma] , (R)^(1/\[Rho]) \[Beta]^(1/\[Rho]) , (R)}];
Print["\[CapitalGamma] <((R) \[Beta]\!\(\*SuperscriptBox[\()\), \(\(1/\[Rho]\)\(\\\ \)\)]\)< R:",\[CapitalGamma] <((R) \[Beta])^(1/\[Rho] )< R];
Print["RIC holds (\[CapitalThorn] < R):",  \[CapitalThorn] <(R)];
Print[" \[Implies] FHWC holds (\[CapitalGamma] < R):",\[CapitalGamma] < R];
m\[Sharp] = (\[HBar]Inf-1)(\[Kappa]MinInf/(1-\[Kappa]MinInf));
\[ScriptC]\[EmptySmallCircle][\[ScriptM]_] := Min[\[ScriptM],m\[Sharp] + \[Kappa]MinInf (\[ScriptM]-m\[Sharp])];
{mMinPlot,mMaxPlot}={0,2};
{cMinPlot,cMaxPlot}={mMinPlot,mMaxPlot};
PFGICFailsRICHoldsPlot = Plot[{\[ScriptC]SuchThat\[DoubleStruckCapitalE]mtp1Equalsmt[m],m,\[ScriptC]\[Digamma]Inf[m],\[ScriptC]\[EmptySmallCircle][m]},{m,mMinPlot,mMaxPlot},PlotRange->All
,PlotStyle->{Red,Green,Blue,Black}
,AxesLabel->{"\[ScriptM]","\[ScriptC]"}];
Print[PFGICFailsRICHolds=Show[PFGICFailsRICHoldsPlot
,Graphics[Text["\[LowerLeftArrow] \!\(\*OverscriptBox[\(\[ScriptC]\), \(_\)]\)(\[ScriptM])",{0.1mMaxPlot,\[ScriptC]\[Digamma]Inf[0.1 mMaxPlot]},{0,-1}]]
,Graphics[Text["\[ScriptC](\[ScriptM])=\[ScriptM]\[RightArrow]",{0.9mMaxPlot,0.9 mMaxPlot},{1,0}]]
,Graphics[Text["\!\(\*SubscriptBox[\(\[DoubleStruckCapitalE]\), \(t\)]\)[\[CapitalDelta] \!\(\*SubscriptBox[\(\[ScriptM]\), \(t + 1\)]\)]=0 \[LowerRightArrow]",{0.4mMaxPlot,\[ScriptC]SuchThat\[DoubleStruckCapitalE]mtp1Equalsmt[0.4 mMaxPlot]},{1,-1}]]
,Graphics[Text["\!\(\*OverscriptBox[\(\[ScriptC]\), \(\[EmptySmallCircle]\)]\)(\[ScriptM]) \[LowerRightArrow]",{0.8 mMaxPlot,\[ScriptC]\[EmptySmallCircle][0.8 mMaxPlot]},{1,-1}]]
]
];


(* Example where PF-GIC, RIC, and FHWC all hold *)
Clear[R,\[Beta],\[CapitalGamma],\[Rho],m\[Sharp],\[ScriptC]\[EmptySmallCircle]];
Print["{R,\[Beta],\[CapitalGamma],\[Rho]}=",{R=1.04,\[Beta]=1/1.02,\[CapitalGamma]=1.02,\[Rho]=2}];
Print["{\[CapitalThorn]=(R\!\(\*SuperscriptBox[\()\), \(1/\[Rho]\)]\) \!\(\*SuperscriptBox[\(\[Beta]\), \(1/\[Rho]\)]\) , \[CapitalGamma] , (R)}=",{(R)^(1/\[Rho]) \[Beta]^(1/\[Rho]) , \[CapitalGamma] , (R)}];
Print["\[CapitalThorn] < \[CapitalGamma] < R:",((R) \[Beta])^(1/\[Rho] )< \[CapitalGamma] < R];
Print["PF-GIC Holds (\[CapitalThorn] < \[CapitalGamma]):" , \[CapitalThorn]< \[CapitalGamma]];
Print["RIC holds (\[CapitalThorn] < R):",  \[CapitalThorn] <(R)];
Print["FHWC holds (\[CapitalGamma] < R):",\[CapitalGamma] < R];
m\[Sharp] = (\[HBar]Inf-1)(\[Kappa]MinInf/(1-\[Kappa]MinInf));
\[ScriptC]\[EmptySmallCircle][\[ScriptM]_] := Min[\[ScriptM],m\[Sharp] + \[Kappa]MinInf (\[ScriptM]-m\[Sharp])];
{mMinPlot,mMaxPlot}={0,2};
{cMinPlot,cMaxPlot}={mMinPlot,mMaxPlot};
PFGICHoldsRICHoldsPlot = Plot[{\[ScriptC]SuchThat\[DoubleStruckCapitalE]mtp1Equalsmt[m],m,\[ScriptC]\[Digamma]Inf[m],\[ScriptC]\[EmptySmallCircle][m]},{m,mMinPlot,mMaxPlot},PlotRange->All
,PlotStyle->{Red,Green,Blue,Black}
,AxesLabel->{"\[ScriptM]","\[ScriptC]"}];
Print[PFGICHoldsRICHolds=Show[PFGICHoldsRICHoldsPlot
,Graphics[Text["\[LowerLeftArrow] \!\(\*OverscriptBox[\(\[ScriptC]\), \(_\)]\)(\[ScriptM])",{0.1mMaxPlot,\[ScriptC]\[Digamma]Inf[0.1 mMaxPlot]},{0,-1}]]
,Graphics[Text["\[ScriptC](\[ScriptM])=\[ScriptM]\[RightArrow]",{0.9mMaxPlot,0.9 mMaxPlot},{1,0}]]
,Graphics[Text["\!\(\*SubscriptBox[\(\[DoubleStruckCapitalE]\), \(t\)]\)[\[CapitalDelta] \!\(\*SubscriptBox[\(\[ScriptM]\), \(t + 1\)]\)]=0 \[LowerRightArrow]",{0.4mMaxPlot,\[ScriptC]SuchThat\[DoubleStruckCapitalE]mtp1Equalsmt[0.4 mMaxPlot]},{1,-1}]]
,Graphics[Text["\!\(\*OverscriptBox[\(\[ScriptC]\), \(\[EmptySmallCircle]\)]\)(\[ScriptM]) \[LowerRightArrow]",{0.95mMaxPlot,\[ScriptC]\[EmptySmallCircle][0.95 mMaxPlot]},{1,-1}]]
]
];


(* Example where PF-GIC holds, FHWC fails, RIC holds, PF-FVAC holds *)
(* Show MPC asymptotes to \[Kappa]MinInf *)
Clear[R,\[Beta],\[CapitalGamma],\[Rho],\[ScriptC]From\[ScriptB],\[ScriptN]Approx,\[ScriptN]Exact];
\[ScriptN]Approx[\[ScriptB]_] :=If[\[ScriptB]+(1/(1-\[ScriptCapitalR]^-1))<=0,1,(Log[(-\[CapitalThorn]Rtn/(1-\[CapitalThorn]Rtn))+(-\[ScriptCapitalR]^-1/(1-\[ScriptCapitalR]^-1))]- Log[\[ScriptB]+(1/(1-\[ScriptCapitalR]^-1))])/Log[\[ScriptCapitalR]]];
\[ScriptN]Exact[\[ScriptB]_] :=nSeek /. FindRoot[b\[Sharp][nSeek] ==\[ScriptB],{nSeek,Max[1,\[ScriptN]Approx[\[ScriptB]]]}];
\[ScriptC]From\[ScriptB][\[ScriptB]_] := \[CapitalThorn]\[CapitalGamma]PF^-\[ScriptN]Exact[\[ScriptB]];
Print["{R,\[Beta],\[CapitalGamma],\[Rho]}=",{R= 1.01,\[Beta] = 0.97,\[CapitalGamma]=1.02,\[Rho]=2}];
Print["\[CapitalThorn]=(R\!\(\*SuperscriptBox[\()\), \(1/\[Rho]\)]\) \!\(\*SuperscriptBox[\(\[Beta]\), \(1/\[Rho]\)]\)=",(R)^(1/\[Rho]) \[Beta]^(1/\[Rho])];
Print["PF-GIC holds (\[CapitalThorn] < \[CapitalGamma]):" , \[CapitalThorn]< \[CapitalGamma]];
Print["RIC holds (\[CapitalThorn] < R):",  \[CapitalThorn] <(R)];
Print["FHWC fails (R < \[CapitalGamma]):", R< \[CapitalGamma]];
Print["PF-FVAC holds \[CapitalThorn]/\[CapitalGamma] < (R/\[CapitalGamma]\!\(\*SuperscriptBox[\()\), \(1/\[Rho]\)]\):",\[CapitalThorn]/\[CapitalGamma]<(R/\[CapitalGamma])^(1/\[Rho])];

{nMin,nMax}={300,500};
KinkPoints=Table[{b\[Sharp][n],c\[Sharp][n]},{n,nMin,nMax}];
KinkPointsFunc=Interpolation[KinkPoints,InterpolationOrder->2];
KinkPointsPlot=ListPlot[KinkPoints,PlotRange->All];
ComparePlots=Show[KinkPointsPlot,Plot[\[ScriptC]From\[ScriptB][\[ScriptB]],{\[ScriptB],b\[Sharp][nMin],b\[Sharp][nMax]}]];
Print[Plot[{KinkPointsFunc'[\[ScriptB]],\[Kappa]MinInf},{\[ScriptB],b\[Sharp][nMin],b\[Sharp][nMax]},PlotRange->All]];


Print["Example where PF-GIC holds, FHWC fails, RIC fails, PF-FVAC Fails"];
Clear[R,\[Beta],\[CapitalGamma],\[Rho],\[ScriptC]From\[ScriptB],\[ScriptN]Approx,\[ScriptN]Exact];
\[ScriptN]Approx[\[ScriptB]_] :=If[\[ScriptB]+(1/(1-\[ScriptCapitalR]^-1))<=0,1,(Log[(-\[CapitalThorn]Rtn/(1-\[CapitalThorn]Rtn))+(-\[ScriptCapitalR]^-1/(1-\[ScriptCapitalR]^-1))]- Log[\[ScriptB]+(1/(1-\[ScriptCapitalR]^-1))])/Log[\[ScriptCapitalR]]];
\[ScriptN]Exact[\[ScriptB]_] :=nSeek /. FindRoot[b\[Sharp][nSeek] ==\[ScriptB],{nSeek,Max[1,\[ScriptN]Approx[\[ScriptB]]]}];
\[ScriptC]From\[ScriptB][\[ScriptB]_] := \[CapitalThorn]\[CapitalGamma]PF^-\[ScriptN]Exact[\[ScriptB]];
Print["{R,\[Beta],\[CapitalGamma],\[Rho]}=",{R=0.98,\[Beta] = 0.99,\[CapitalGamma]=1.0,\[Rho]=2}];
Print["{R,\[Beta],\[CapitalGamma],\[Rho]}=",{R,\[Beta],\[CapitalGamma],\[Rho]}={0.98,1.00,0.99,2}];
Print["{\[CapitalThorn]=(R\!\(\*SuperscriptBox[\()\), \(1/\[Rho]\)]\) \!\(\*SuperscriptBox[\(\[Beta]\), \(1/\[Rho]\)]\) , \[CapitalGamma] , (R)}=",{(R)^(1/\[Rho]) \[Beta]^(1/\[Rho]) , \[CapitalGamma] , (R)}];
Print["PF-GIC holds (\[CapitalThorn] < \[CapitalGamma]):" , \[CapitalThorn]< \[CapitalGamma]];
Print["RIC fails (R < \[CapitalThorn]):",  (R) <\[CapitalThorn]];
Print["FHWC fails (R < \[CapitalGamma]):", R< \[CapitalGamma]];
Print["PF-FVAC fails (R/\[CapitalGamma]\!\(\*SuperscriptBox[\()\), \(1/\[Rho]\)]\) < \[CapitalThorn]/\[CapitalGamma] (same as 1 < \[Beta] \!\(\*SuperscriptBox[\(\[CapitalGamma]\), \(1 - \[Rho]\)]\)):",1 <\[Beta] \[CapitalGamma]^(1-\[Rho])];
{nMin,nMax}={0,300};
KinkPoints=Table[{b\[Sharp][n],c\[Sharp][n]},{n,nMin,nMax}];
KinkPointsFunc=Interpolation[KinkPoints,InterpolationOrder->2];
KinkPointsPlot=ListPlot[KinkPoints,PlotRange->All];
cfbPlot=Plot[{\[ScriptC]From\[ScriptB][\[ScriptB]]},{\[ScriptB],b\[Sharp][nMin],b\[Sharp][nMax]}];
PFGICHoldsFHWCFailsRICFails=Show[KinkPointsPlot,cfbPlot
,AxesLabel->{"\[ScriptM]","\[ScriptC]"}
];
CDToHomeDir;
ExportFigs["PFGICHoldsFHWCFailsRICFails"];
Print[Plot[{KinkPointsFunc'[\[ScriptB]]},{\[ScriptB],b\[Sharp][nMax-200],b\[Sharp][nMax]},PlotRange->All]];


Print["Example where PF-GIC holds, FHWC fails, RIC fails, PF-FVAC holds"];
Clear[R,\[Beta],\[CapitalGamma],\[Rho],\[ScriptC]From\[ScriptB],\[ScriptN]Approx,\[ScriptN]Exact];
\[ScriptN]Approx[\[ScriptB]_] :=If[\[ScriptB]+(1/(1-\[ScriptCapitalR]^-1))<=0,1,(Log[(-\[CapitalThorn]Rtn/(1-\[CapitalThorn]Rtn))+(-\[ScriptCapitalR]^-1/(1-\[ScriptCapitalR]^-1))]- Log[\[ScriptB]+(1/(1-\[ScriptCapitalR]^-1))])/Log[\[ScriptCapitalR]]];
\[ScriptN]Exact[\[ScriptB]_] :=nSeek /. FindRoot[b\[Sharp][nSeek] ==\[ScriptB],{nSeek,Max[1,\[ScriptN]Approx[\[ScriptB]]]}];
\[ScriptC]From\[ScriptB][\[ScriptB]_] := \[CapitalThorn]\[CapitalGamma]PF^-\[ScriptN]Exact[\[ScriptB]];
Print["{R,\[Beta],\[CapitalGamma],\[Rho]}=",{R=0.98,\[Beta] = 0.99,\[CapitalGamma]=1.0,\[Rho]=2}];
Print["{\[CapitalThorn]=(R\!\(\*SuperscriptBox[\()\), \(1/\[Rho]\)]\) \!\(\*SuperscriptBox[\(\[Beta]\), \(1/\[Rho]\)]\) , \[CapitalGamma] , (R)}=",{(R)^(1/\[Rho]) \[Beta]^(1/\[Rho]) , \[CapitalGamma] , (R)}];
Print["PF-GIC holds (\[CapitalThorn] < \[CapitalGamma]):" , \[CapitalThorn]< \[CapitalGamma]];
Print["RIC fails (R < \[CapitalThorn]):",  (R) <\[CapitalThorn]];
Print["FHWC fails (R < \[CapitalGamma]):", R< \[CapitalGamma]];
Print["PF-FVAC holds  \[CapitalThorn]/\[CapitalGamma] < (R/\[CapitalGamma])^(1/\[Rho])):",\[Beta] \[CapitalGamma]^(1-\[Rho])<1];
{nMin,nMax}={0,300};
KinkPoints=Table[{b\[Sharp][n],c\[Sharp][n]},{n,nMin,nMax}];
KinkPointsFunc=Interpolation[KinkPoints,InterpolationOrder->2];
KinkPointsPlot=ListPlot[KinkPoints,PlotRange->All];
cfbPlot=Plot[{\[ScriptC]From\[ScriptB][\[ScriptB]]},{\[ScriptB],b\[Sharp][nMin],b\[Sharp][nMax]}];
PFGICHoldsFHWCFailsRICFailsPFVACHolds=Show[KinkPointsPlot,cfbPlot
,AxesLabel->{"\[ScriptM]","\[ScriptC]"}
];
CDToHomeDir;
ExportFigs["PFGICHoldsFHWCFailsRICPFVACHoldsFails"];
Print[Plot[{KinkPointsFunc'[\[ScriptB]]},{\[ScriptB],b\[Sharp][nMax-200],b\[Sharp][nMax]},PlotRange->All]];




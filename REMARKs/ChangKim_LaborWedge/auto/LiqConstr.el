(TeX-add-style-hook
 "ChangKim-LaborWedge"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("datetime2" "en-US")))
   (TeX-run-style-hooks
    "./econtexRoot"
    "./econtexPaths"
    "Sections/Intro"
    "Sections/Model"
    "vmargin"
    "float"
    "datetime2"
    "grfext")
   (TeX-add-symbols
    "texname"
    "versn"
    "ifVerbatimWrite"
    "wAlt"
    "sConst")
   (LaTeX-add-labels
    "sec:Intro"
    "sec: Goal"
    "fig:goal"
    "sec: Error"
    "sec: The Problem"
    "fig:mrs_prod"
    "fig:wage_hours"
    "sec:Model"
    "sec:Setup"
    "sec:Equilibrium"
    "sec:QuantAnalysis"
    "sec:Calibration"
    "fig:table_1"
    "sec:CrossSectional"
    "fig:table_2"
    "fig:figure_3"
    "sec:CyclicalProperties"
    "fig:table_3"
    "fig:table_4"
    "sec:Notes"
    "sec:Results"
    "fig:agg_savings"
    "fig:good_consumption"
    "fig:bad_consumption"
    "fig:labor_good"
    "fig:labor_bad")
   (LaTeX-add-environments
    "Private"
    "defn"
    "theorem"
    "lemma"
    "corollary"
    "prop")
   (LaTeX-add-bibliographies
    "ChangKim-LaborWedge-Add")
   (LaTeX-add-lengths
    "TableWidth"))
 :latex)


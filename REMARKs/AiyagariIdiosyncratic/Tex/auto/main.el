(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("babel" "english")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "Tables/Table_Parameters"
    "Tables/Table_SavingRate"
    "article"
    "art10"
    "inputenc"
    "babel"
    "graphicx"
    "hyperref"
    "appendix"
    "mathtools"
    "float"
    "blindtext"
    "subfiles"
    "verbatim"
    "apacite")
   (TeX-add-symbols
    "EqDir"
    "RefDir")
   (LaTeX-add-labels
    "table:2"
    "figure:1")
   (LaTeX-add-bibliographies
    "References/Aiyagari1994"))
 :latex)


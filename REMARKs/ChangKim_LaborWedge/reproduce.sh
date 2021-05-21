# Run the noetbook and create figures
ipython code.py

# compile latex file
pdflatex ChangKim-LaborWedge.tex
bibtex ChangKim-LaborWedge
pdflatex ChangKim-LaborWedge.tex
pdflatex ChangKim-LaborWedge.tex
rm ChangKim-LaborWedge.aux ChangKim-LaborWedge.bbl ChangKim-LaborWedge.blg ChangKim-LaborWedge.dep ChangKim-LaborWedge.log ChangKim-LaborWedge.out

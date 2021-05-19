# Run the notebook and create figures
ipython ./Code/Consumption.py
ipython ./Code/AssetWealth.py

# compile latex file
pdflatex LiqConstr.tex
bibtex LiqConstr
pdflatex LiqConstr.tex
pdflatex LiqConstr.tex
rm LiqConstr.aux LiqConstr.bbl LiqConstr.blg LiqConstr.dep LiqConstr.log LiqConstr.out

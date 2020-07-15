#!/bin/bash

python3 -m pip install --ignore-installed -r requirements.txt

ipython Aiyagari1994QJE.py
# python ./do_all.py #create and save figures and tables

cd Tex
pdflatex main.tex #creates the main paper and aux file
bibtex main.aux  #run bibtex to run aux to create bbl file
pdflatex main.tex #rerun latex with the bbl file

cd Slides
pdflatex IndividualsProblem.tex

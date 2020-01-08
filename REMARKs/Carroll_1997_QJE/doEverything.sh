#!/bin/bash

scriptDir="$(realpath $(dirname "$0"))" # get the path to this script itself

sudo echo 'Authorizing sudo.'

python ./Carroll_1997_QJE.py #save figures and tables

cd Paper
pdflatex main.tex #creates the main paper and aux file
bibtex main.aux  #run bibtex to run aux to create bbl file
pdflatex main.tex #rerun latex with the bbl file

cd ..
cd Slides
pdflatex slides.tex

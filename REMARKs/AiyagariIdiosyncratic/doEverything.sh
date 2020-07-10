#!/bin/bash

pip install altair

pip install tabulate

scriptDir="$(realpath $(dirname "$0"))" # get the path to this script itself

sudo echo 'Authorizing sudo.'

python ./do_all.py #create and save figures and tables

cd Tex
pdflatex main.tex #creates the main paper and aux file
bibtex main.aux  #run bibtex to run aux to create bbl file
pdflatex main.tex #rerun latex with the bbl file

cd Slides
pdflatex IndividualsProblem.tex

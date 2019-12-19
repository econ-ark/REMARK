#!/bin/bash

scriptDir="$(realpath $(dirname "$0"))" # get the path to this script itself

sudo echo 'Authorizing sudo.'

python ./Carroll_1997_RemARK.py #save figures and tables

cd Paper
xelatex main.tex #creates the main paper and aux file
bibtex main.aux  #run bibtex to run aux to create bbl file
xelatex main.tex #rerun latex with the bbl file

cd ..
cd Slides

chmod u+x make_slides.sh #This is to grant permission to making the slides

./make_slides.sh #create slides

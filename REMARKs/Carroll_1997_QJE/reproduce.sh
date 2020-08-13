#!/bin/bash
# tabulate needs to be installed for writing tex tables
python3 -m pip install tabulate

# Go to the ipython notebook/script directory to execute the script
# and create figures and tables.
cd Code/Python
ipython Carroll_1997_QJE.py

# Navigate to Paper directory to build paper
cd ../../Paper
pdflatex main.tex #creates the main paper and aux file
bibtex main.aux  #run bibtex to run aux to create bbl file
pdflatex main.tex #rerun latex with the bbl file

# Navigate to Slides directory to build the slides
cd ../Slides
pdflatex slides.tex

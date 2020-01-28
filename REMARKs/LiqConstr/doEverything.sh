#!/bin/bash

./do_all_code.sh

cd Paper-Original

pdflatex LiqConstr
bibtex   LiqConstr
pdflatex LiqConstr
pdflatex LiqConstr



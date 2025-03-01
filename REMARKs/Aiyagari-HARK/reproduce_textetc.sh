#!/bin/bash

pdflatex --output-directory=latex REMARK-starter-example.tex
bibtex latex/REMARK-starter-example
pdflatex --output-directory=latex REMARK-starter-example.tex
pdflatex --output-directory=latex REMARK-starter-example.tex

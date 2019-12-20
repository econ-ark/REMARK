#!/bin/bash


scriptDir="$(realpath $(dirname "$0"))" # Parent directory, e.g. BufferStockTheory-make
# scriptDir=~/Papers/BST/BST-make
baseName=$(basename $(dirname "$scriptDir")) # Name of grandparent directory, e.g. BufferStockTheory

SharedDir="$(realpath "$scriptDir/../$baseName-Shared")" # e.g., BufferStockTheory-Shared

toolsDir=/Methods/Tools/Scripts # Extra tools

journal=TheOnion
letter=Submit

cd $scriptDir/../$baseName-Shared/Private/$journal


pdflatex "$letter"
bibtex   "$letter"
pdflatex "$letter"
pdflatex "$letter"
pdflatex "$letter"


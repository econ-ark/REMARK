#!/bin/bash
# Creates all the figures for the paper, then the journal interactions, then makes an archive
# and finally (optionally) updates the GitHub version

scriptDir="$(realpath $(dirname "$0"))" # Parent directory, e.g. BufferStockTheory-make 
baseName=$(basename $(dirname "$scriptDir")) # Name of grandparent directory, e.g. BufferStockTheory

SharedDir="$(realpath "$scriptDir/../$baseName-Shared")" # e.g., BufferStockTheory-Shared

toolsDir=/Methods/Tools/Scripts # Extra tools

cd "$scriptDir"

./makeFiguresHARK.sh
#./makeJournalStuff.sh
./makePDF-Shareable.sh
./makePublic.sh ~/Papers/CGMPort update

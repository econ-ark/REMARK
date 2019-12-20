#!/bin/bash
# Unix bash script to construct figures using Econ-ARK/HARK toolkit

scriptDir="$(realpath $(dirname "$0"))" # Parent directory, e.g. BufferStockTheory-make
# scriptDir=~/Papers/BST/BST-make
baseName=$(basename $(dirname "$scriptDir")) # Name of grandparent directory, e.g. BufferStockTheory

SharedDir="$(realpath "$scriptDir/../$baseName-Shared")" # e.g., BufferStockTheory-Shared

toolsDir=/Methods/Tools/Scripts # Extra tools

scriptDir="$(dirname "$0")" 
cd $scriptDir/../"$baseName"-Shared/Code/Python

#ipython ./CGM_REMARK.py

ipython ./do_ALL.py
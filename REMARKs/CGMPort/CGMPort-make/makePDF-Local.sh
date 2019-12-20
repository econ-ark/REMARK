#!/bin/bash

scriptDir="$(realpath $(dirname "$0"))" # Parent directory, e.g. BufferStockTheory-make
# scriptDir=~/Papers/BST/BST-make
baseName=$(basename $(dirname "$scriptDir")) # Name of grandparent directory, e.g. BufferStockTheory

SharedDir="$(realpath "$scriptDir/../$baseName-Shared")" # e.g., BufferStockTheory-Shared

toolsDir=/Methods/Tools/Scripts # Extra tools

cd "$scriptDir"

cmd="$toolsDir/makePDF-Local.sh `realpath ../$baseName-Shared` CGMPort"
echo "$cmd"
eval "$cmd"

cmd="$toolsDir/makePDF-Local.sh `realpath ../$baseName-Shared/Slides` CGMPort-Slides"
echo "$cmd"
eval "$cmd"



#!/bin/sh
# This script assumes that its grandparent directory is the name of the paper (baseName)
# and the contents of the shared repo are in a subdirectory (named "-Shared")
# of the grandparent directory

scriptDir="$(realpath $(dirname "$0"))" # Parent directory, e.g. BufferStockTheory-make 
baseName=$(basename $(dirname "$scriptDir")) # Name of grandparent directory, e.g. BufferStockTheory

SharedDir="$(realpath "$scriptDir/../$baseName-Shared")" # e.g., BufferStockTheory-Shared

toolsDir=/Methods/Tools/Scripts # Extra tools

$toolsDir/makePDF-Shareable.sh `realpath $SharedDir`        CGMPort
$toolsDir/makePDF-Shareable.sh `realpath $SharedDir/Slides` CGMPort-Slides


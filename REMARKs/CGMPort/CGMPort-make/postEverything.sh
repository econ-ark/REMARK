#!/bin/bash
scriptDir="$(realpath $(dirname "$0"))" # Parent directory, e.g. BufferStockTheory-make 
baseName=$(basename $(dirname "$scriptDir")) # Name of grandparent directory, e.g. BufferStockTheory

SharedDir="$(realpath "$scriptDir/../$baseName-Shared")" # e.g., BufferStockTheory-Shared

toolsDir=/Methods/Tools/Scripts # Extra tools

cd $scriptDir/../$baseName-Shared

git fetch
git status

echo '' ; echo ''
echo 'To post your changes, from a shell in the -Shared directory, please do:'
echo ''
echo 'git add .'
echo 'git commit -m [commit message]'
echo 'git push'
echo ''
echo ''
echo 'Hit return when done, C-c to abort'
read answer

cd $scriptDir/../$baseName-Public

git fetch
git status

echo '' ; echo ''
echo 'To post your changes, from a shell in the -Public directory, please do:'
echo ''
echo 'git add .'
echo 'git commit -m [commit message]'
echo 'git push'
echo ''
echo ''
echo 'Hit return when done, C-c to abort'
read answer



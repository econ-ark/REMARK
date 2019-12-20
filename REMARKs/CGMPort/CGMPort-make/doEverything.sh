#!/bin/bash

scriptDir="$(realpath $(dirname "$0"))" # get the path to this script itself

sudo echo 'Authorizing sudo.'

cd $scriptDir

./makeEverything.sh
./postEverything.sh
./makePDF-Local.sh # Fixes the fact that the makeEverything code leaves the LaTeX directory in its "Portable" rather than "Local" state

#!/bin/bash

# Construct [name]-Public from repo

if [ $# -ne 2 ]; then
    echo "usage:   ${0##*/} <rootpath> create|update "
    echo "example: ${0##*/} ~/Papers/CGMPort-Shared update"
    exit 1
fi

pathLocal=`realpath $1`
option=$2

scriptDir="$(realpath $(dirname "$0"))" # get the path to this script itself

# scriptDir=~/Papers/BST/BST-make ; option=create ; pathLocal=~/Papers/BST
scriptRoot="$(realpath "$scriptDir"/..)" # Assume the scriptRoot is one level up
nameRoot="$(basename  "$scriptRoot")" # Assume the base name of the project is the name of the root directory

if [ "$option" == "create" ]; then # If dir does not exist, create it
    repo_local=$pathLocal/$nameRoot-Public
    if [ ! -e $repo_local ]; then
	mkdir -p $repo_local
    else
	msg='Local repo directory already exists; hit return to proceed anyway, C-c to stop'
	echo $msg
	read answer 
    fi
fi

if [ "$option" == "create" ]; then # create a new directory 
    mkdir -p $pathLocal/$nameRoot-Public # make sure the target exists
    rsync -azhv --delete-excluded --inplace  --exclude="*.git" --exclude="Private" --force $scriptRoot/$nameRoot-Shared/ $pathLocal/$nameRoot-Public
else # refresh an existing directory
    rsync -azhv                   --inplace  --exclude="*.git" --exclude="Private" --force $scriptRoot/$nameRoot-Shared/ $pathLocal/$nameRoot-Public
fi

# strip everything between begin{Private} and end{Private}, remove all lines labeled PrivateMsg, and remove all comments
cd $pathLocal/$nameRoot-Public
for f in $(find . -name '*.tex')
do
    sed '/begin{Private}/,/end{Private}/d' $f > $f-tmp # Delete everything in a {Private} environment 
    grep -v '%Msg' $f-tmp > $f-tmp2                # Delete all lines marked as %Msg
    sed '/%%/d' $f-tmp2 > $f                           # Remove all comments beginning with %%
    rm $f-tmp $f-tmp2                                  # Clean up
done


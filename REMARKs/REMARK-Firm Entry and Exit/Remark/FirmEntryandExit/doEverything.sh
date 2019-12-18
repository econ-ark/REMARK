#!/bin/bash

scriptDir="$(realpath $(dirname "$0"))"  # Parent directory, e.g. FirmEntryandExit
# scriptDir=~/FirmEntryandExit

sudo echo 'Authorizing sudo.'

python ./do_all.py
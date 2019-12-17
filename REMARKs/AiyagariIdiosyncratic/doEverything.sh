#!/bin/bash

scriptDir="$(realpath $(dirname "$0"))"  # Parent directory, e.g. AiyagariIdiosyncratic
# scriptDir=~/AiyagariIdiosyncratic

sudo echo 'Authorizing sudo.'

python ./do_all.py
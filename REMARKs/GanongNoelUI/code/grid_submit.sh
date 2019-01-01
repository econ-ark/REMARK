#!/bin/bash
#$ -N python
#$ -q short
#$ -j y
#$ -cwd
/usr/bin/python estimate_models.py

#!/bin/sh
# This script is used to make Python wheel.
echo "Make wheel archive for BMLiNGAM."
echo "Workdir = /home/jovyan/work"

mkdir /home/jovyan/work/dist
python setup.py bdist_wheel --universal

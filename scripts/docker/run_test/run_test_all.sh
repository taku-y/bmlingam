#!/bin/sh
# This script execute processes for running test of BMLiNGAM.
# This script will run in a Docker container.
# It is assumed that /home/jovyan/work is mounted on the root directory of the
# bmlingam repo.

echo $MPLBACKEND

# # 1/3 Make wheel of BMLiNGAM
sh scripts/docker/make_wheel.sh

# # 2/3 Install BMLiNGAM
cd dist
WHEEL=`ls .`
echo $WHEEL
pip install $WHEEL
cd ..

# # Check if bmlingam was succesfully installed
# ipython -c 'import matplotlib; matplotlib.use("Agg"); import bmlingam; print(bmlingam)'

# 3/3 Run test
nosetests bmlingam.tests.test_utils
nosetests bmlingam.tests.test_infer_causality

#!/bin/bash
# This script execute processes for running test of BMLiNGAM.
# This script will run in a Docker container.
# It is assumed that /home/jovyan/work is mounted on the root directory of the
# bmlingam repo.

# Switch environment
if [ $# -ge 1 ]; then
    if [ $1 = "py27" ]; then
        source activate py27
    fi
fi

# # 1/3 Make wheel of BMLiNGAM
sh scripts/docker/make_wheel.sh

# 2/3 Install BMLiNGAM
cd dist
WHEEL=`ls .`
echo $WHEEL
pip install --user $WHEEL
cd ..

# 3/3 Run test
pip install --user nose
~/.local/bin/nosetests -v bmlingam.tests.test_utils
~/.local/bin/nosetests -v bmlingam.tests.test_infer_causality
~/.local/bin/nosetests -v bmlingam.tests.test_cli

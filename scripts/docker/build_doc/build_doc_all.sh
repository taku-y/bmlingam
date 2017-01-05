#!/bin/sh
# This script execute processes for building document.
# This script will run in a Docker container.
# It is assumed that /home/jovyan/work is mounted on the root directory of the
# bmlingam repo.

# 1/2 Make wheel of BMLiNGAM
sh scripts/docker/make_wheel.sh

# 2/2 Build document
cd doc
make html
cd ..

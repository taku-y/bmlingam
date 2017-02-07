#!/bin/sh
# This script is used to run commands in tutorial on Docker.
# A wheel archive will be created under dist directory.
export PROJROOT=$(pwd)/..
docker run --rm -v $PROJROOT:/home/jovyan/work --name bml_run_tutorial \
    bml_run_tutorial bash scripts/docker/run_tutorial/run_tutorial_all.sh

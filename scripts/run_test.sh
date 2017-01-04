#!/bin/sh
# This script is used to run tests on Docker.
# A wheel archive will be created under dist directory.
export PROJROOT=$(pwd)/..
docker run --rm -v $PROJROOT:/home/jovyan/work --name bml_run_test bml_run_test sh scripts/docker/run_test/run_test_all.sh
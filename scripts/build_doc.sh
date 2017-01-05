#!/bin/sh
# This script is used to build document on Docker.
# A wheel archive will be created under dist directory.
export PROJROOT=$(pwd)/..
docker run --rm -v $PROJROOT:/home/jovyan/work --name bml_run_test bml_build_doc sh scripts/docker/build_doc/build_doc_all.sh
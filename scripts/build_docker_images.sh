#!/bin/sh
# This script is used to build Docker images.
# You needs to run this at bmlingam/scripts.
pushd ./docker/run_test
docker build -t bml_run_test .
popd

pushd ./docker/build_doc
docker build -t bml_build_doc .
popd

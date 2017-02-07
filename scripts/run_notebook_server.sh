#!/bin/sh
# This script is used to run commands in tutorial on Docker.
# A wheel archive will be created under dist directory.
docker run --rm -v $HOME:/home/jovyan/work --name bml_nbserver \
    -p $1:8888 bml_run_tutorial
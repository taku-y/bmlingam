#!/bin/sh

if [ $# -eq 0 ]; then
    echo "Docker image name is required"
else
    docker run --rm -it -v "$(pwd)/..":/root $1 /bin/bash -c \
        "cd /root; \
         nosetest --help"
fi


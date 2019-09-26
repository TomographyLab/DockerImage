#!/bin/bash
export SRC_FOLD=${PWD}/src/
export NB_FOLD=${PWD}/notebooks/

docker system prune -f
nvidia-docker build -t occiput:test -f config/occiput.Dockerfile .
nvidia-docker run --runtime=nvidia --rm --init -it -v $NB_FOLD:/home/occiput/notebooks -v $SRC_FOLD:/home/occiput/code -p 8888:8888 -p 10000:8080 occiput:test

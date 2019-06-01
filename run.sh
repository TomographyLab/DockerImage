#!/bin/bash
export SRC_FOLD=${PWD}/src/
export NB_FOLD=${PWD}/notebooks/
#export DATA_FOLD=<custom host path>

docker system prune -f
nvidia-docker build -t occiput:devel -f config/occiput.Dockerfile .
nvidia-docker run --runtime=nvidia \
                  --rm \
                  --init \
                  -it \
                  -v $NB_FOLD:/home/occiput/notebooks \
                  -v $SRC_FOLD:/home/occiput/code \
                  #-v $DATA_FOLD:/home/occiput/data \
                  -p 8888:8888 \
                  -p 10000:8080 \
                  occiput:devel
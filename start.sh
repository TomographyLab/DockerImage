#!/bin/bash
export SRC_FOLD=${PWD}/source/
export NB_FOLD=${PWD}/notebooks/
export DATA_FOLD=/media/MAXTOR/
pycharm=$(which pycharm-professional)
atom=$(which atom)

docker system prune -f
echo winter-1989 | sudo -S baobab /var/lib/docker/ & 
$pycharm $SRC_FOLD/tomolab/ &
#$atom $SRC_FOLD/tomolab/ &
cd ./dockerImages/occiput/
sh run.sh

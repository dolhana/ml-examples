#! /usr/bin/env bash

set -x

BASE_DIR=$(realpath $(dirname $0))

docker build -t ddpg .
exec docker run -ti --name ddpg -p 8888:8888 -p 6006:6006 -v $BASE_DIR:/notebooks/work ddpg

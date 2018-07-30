#! /usr/bin/env bash

set -e

BASE_DIR=$(realpath $(dirname $0))

docker build -t ddpg .
CONTAINER_ID=$(docker run --name ddpg -d --rm \
    -p 8888:8888 -p 6006:6006 \
    -v $BASE_DIR:/notebooks/work \
    ddpg)

echo 'ddpg container: ' $CONTAINER_ID

exec docker logs -f -t $CONTAINER_ID

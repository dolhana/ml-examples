#! /usr/bin/env bash

set -e

BASE_DIR=$(realpath $(dirname $0))
IMAGE_NAME=ddpg-gpu

echo "Building the docker image ${IMAGE_NAME}"
docker build -t $IMAGE_NAME -f ${BASE_DIR}/Dockerfile-gpu ${BASE_DIR}

echo "Starting the container ${IMAGE_NAME}"
CONTAINER_ID=$(docker run --runtime=nvidia --name=$IMAGE_NAME -d --rm \
    -p 8888:8888 -p 6006:6006 -p 7007:7007 \
    -v ml-examples:/notebooks/work \
    $IMAGE_NAME)

echo "ddpg container: $CONTAINER_ID"

exec docker logs -f -t --since 5m $CONTAINER_ID

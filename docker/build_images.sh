#!/bin/bash

set -ex

# Image name from command line if given
MINDMAP_DEPS_IMAGE_NAME="mindmap_deps"
[[ $# == 1 ]] && MINDMAP_DEPS_IMAGE_NAME=$1

# Build the nvblox dependencies on cuda 11.8 + ubuntu22
NVBLOX_DEPS_IMAGE_NAME=nvblox_core_deps_cuda118_ubuntu22
echo "Building $NVBLOX_DEPS_IMAGE_NAME."
docker build --network=host \
       --build-arg BASE_IMAGE=nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04 \
       -t $NVBLOX_DEPS_IMAGE_NAME \
    -f submodules/nvblox/docker/Dockerfile.deps submodules/nvblox

# Then add the mindmap dependencies on top
echo "Building $MINDMAP_DEPS_IMAGE_NAME."
docker build --network=host -t $MINDMAP_DEPS_IMAGE_NAME . -f docker/Dockerfile.mindmap_deps \
    --build-arg BASE_IMAGE=${NVBLOX_DEPS_IMAGE_NAME} --progress=plain

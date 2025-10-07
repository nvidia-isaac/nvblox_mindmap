#!/bin/bash
set -ex

WORKSPACE_DIR="/workspaces/mindmap"

# Show cache stats before build
ccache --show-stats

################################################################
# Figure out cuda architecture to build for
################################################################

echo "CMAKE_CUDA_ARCHITECTURES: $CMAKE_CUDA_ARCHITECTURES"
# if env CMAKE_CUDA_ARCHITECTURES is not set, we detect compute cuda architecture (compute capability) by querying nvidia-smi
[[ -z "$CMAKE_CUDA_ARCHITECTURES" ]] &&  CMAKE_CUDA_ARCHITECTURES=$(nvidia-smi --query-gpu=compute_cap --format=csv | tail -1 | tr -d '.')
echo "Building for CUDA architecture[s]: $CMAKE_CUDA_ARCHITECTURES"

################################################################
# Build nvblox
################################################################

cd $WORKSPACE_DIR/submodules/nvblox
mkdir -p build && cd build
# Features array size of 768 is set to support RadioV25B.
cmake .. -DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES -DNVBLOX_FEATURE_ARRAY_NUM_ELEMENTS=768
make -j$(nproc) py_nvblox

################################################################
# Install nvblox_torch
################################################################

# Activate the virtual environment.
. /opt/venv/bin/activate
# Upgrade pip to the latest version (needed for installing toml packages as editable packages).
python3 -m pip install --ignore-installed --upgrade pip
# Install the nvblox_torch pip package.
pip install -e $WORKSPACE_DIR/submodules/nvblox/nvblox_torch

# Show cache stats after build
ccache --show-stats

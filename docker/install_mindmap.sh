#!/bin/bash
set -ex

WORKSPACE_DIR="/workspaces/mindmap"

################################################################
# Install mindmap
################################################################

# Activate the virtual environment.
. /opt/venv/bin/activate
# Upgrade pip to the latest version (needed for installing toml packages as editable packages).
python3 -m pip install --ignore-installed --upgrade pip
# Install the mindmap pip package.
pip install -e $WORKSPACE_DIR

######################a#########################################
# Pre-download feature models to avoid race conditions and
#  "Too Many Requests" HTTP errors.
################################################################

python3 -c 'from mindmap.image_processing.feature_extraction import ClipResNet50FpnFeatureExtractor as fe; fe.load_model()'
python3 -c 'from mindmap.image_processing.feature_extraction import RadioV25BFeatureExtractor as fe; fe.load_model()'
python3 -c 'from mindmap.image_processing.feature_extraction import DinoV2Vits14FeatureExtractor as fe; fe.load_model()'

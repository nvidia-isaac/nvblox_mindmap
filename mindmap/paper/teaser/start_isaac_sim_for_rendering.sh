#!/bin/bash

KIT_FILE_ROOT_DIR="/workspaces/mindmap/submodules/IsaacLab/apps/"
ORIGINAL_KIT_FILE_PATH="$KIT_FILE_ROOT_DIR/isaaclab.python.rendering.kit"
MODIFIED_KIT_FILE_PATH="$KIT_FILE_ROOT_DIR/isaaclab.python.rendering.syn.kit"

# Copy the kit file
cp $ORIGINAL_KIT_FILE_PATH $MODIFIED_KIT_FILE_PATH

# Add the synthetic recorder to the kit file
sed -i '25 a\ "isaacsim.replicator.synthetic_recorder" = {}' $MODIFIED_KIT_FILE_PATH

# Open Isaac Sim
isaacsim $MODIFIED_KIT_FILE_PATH

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import os

from mindmap.cli.args import ClosedLoopAppArgs, DataGenAppArgs, TrainingAppArgs
from mindmap.data_loading.data_types import DataType
from mindmap.image_processing.feature_extraction import FeatureExtractorType
from mindmap.tasks.tasks import Tasks
import mindmap_osmo
from mindmap_osmo.workflow_utils.workflow_types import OsmoTaskType, OsmoWorkflowType, PlatformType

#### WORKFLOW ARG DEFAULTS ####

# MEMORY
DEFAULT_MEMORY_GB = {
    OsmoTaskType.EVAL.name: 30,
    OsmoTaskType.DATAGEN.name: 100,
    OsmoTaskType.TRAINING.name: 250,
}

# STORAGE
MIN_STORAGE_GB = 10
EVAL_STORAGE_GB = 10
DEMO_SIZE_GB = {
    # CUBES/MUG/DRILL_IN_BOX/STICK_IN_BIN: RGBD
    "mindmap_mug_rgbd_x_1cam": 0.2,
    "mindmap_mug_rgbd_x_2cam": 0.2,
    "mindmap_cubes_rgbd_x_1cam": 0.2,
    "mindmap_cubes_rgbd_x_2cam": 0.2,
    "mindmap_drill_rgbd_x_1cam": 0.2,
    "mindmap_drill_rgbd_x_2cam": 0.2,
    "mindmap_stick_rgbd_x_1cam": 0.2,
    "mindmap_stick_rgbd_x_2cam": 0.4,
    # CUBES: MESH
    "mindmap_cubes_rgbdmesh_clip_1cam": 0.2,
    "mindmap_cubes_mesh_clip_1cam": 0.2,
    "mindmap_cubes_mesh_clip_2cam": 0.4,
    "mindmap_cubes_rgbdmesh_radioB_1cam": 1,
    "mindmap_cubes_mesh_radioB_1cam": 1.3,
    "mindmap_cubes_mesh_radioB_2cam": 2.6,
    # MUG: MESH
    "mindmap_mug_rgbdmesh_clip_1cam": 2,
    "mindmap_mug_mesh_clip_1cam": 2,
    "mindmap_mug_mesh_clip_2cam": 2,
    "mindmap_mug_rgbdmesh_radioB_1cam": 5,
    "mindmap_mug_mesh_radioB_1cam": 4,
    "mindmap_mug_mesh_radioB_2cam": 8,
    # DRILL_IN_BOX: MESH
    "mindmap_drill_mesh_radioB_1cam": 8,
    "mindmap_drill_rgbdmesh_radioB_1cam": 8,
    # STICK_IN_BIN: MESH
    "mindmap_stick_mesh_radioB_1cam": 9,
    "mindmap_stick_rgbdmesh_radioB_1cam": 9,
}
MAX_STORAGE_ON_PLATFORM_GB = {
    PlatformType.OVX_L40: 6333,
    PlatformType.OVX_L40S: 3163,
    PlatformType.DGX_H100: 892,
}

# NUM_GPUS
DEFAULT_NUM_GPUS = {
    OsmoTaskType.EVAL.name: 1,
    OsmoTaskType.DATAGEN.name: 1,
    OsmoTaskType.TRAINING.name: 2,
}

# NUM_CPUS
DEFAULT_NUM_CPUS = {
    OsmoTaskType.EVAL.name: 20,
    OsmoTaskType.DATAGEN.name: 20,
    OsmoTaskType.TRAINING.name: 20,
}

# PLATFORM
DEFAULT_PLATFORM = {
    OsmoWorkflowType.EVAL.name: PlatformType.OVX_L40S,
    OsmoWorkflowType.DATAGEN.name: PlatformType.OVX_L40S,
    OsmoWorkflowType.TRAINING.name: PlatformType.DGX_H100,
    OsmoWorkflowType.TRAIN_AND_EVAL.name: PlatformType.OVX_L40S,
    OsmoWorkflowType.E2E.name: PlatformType.OVX_L40S,
}
PLATFORM_TO_POOL = {
    PlatformType.OVX_L40: "PLACEHOLDER_POOL_NAME_L40",
    PlatformType.OVX_L40S: "PLACEHOLDER_POOL_NAME_L40S",
    PlatformType.DGX_H100: "PLACEHOLDER_POOL_NAME_H100",
}

#### INFERRED ARGS ####

# DATASET NAMES
CHECKPOINT_DATASET_NAME = "mindmap_checkpoints"
HDF5_DATASET_NAME = "mindmap_hdf5"
TASK_TYPE_TO_DATASET_NAME = {
    Tasks.CUBE_STACKING.name: "cubes",
    Tasks.MUG_IN_DRAWER.name: "mug",
    Tasks.DRILL_IN_BOX.name: "drill",
    Tasks.STICK_IN_BIN.name: "stick",
}
DATA_TYPE_TO_DATASET_NAME = {
    DataType.MESH.name: "mesh",
    DataType.RGBD_AND_MESH.name: "rgbdmesh",
    DataType.RGBD.name: "rgbd",
}
FEATURE_TYPE_TO_DATASET_NAME = {
    FeatureExtractorType.CLIP_RESNET50_FPN.name: "clip",
    FeatureExtractorType.RADIO_V25_B.name: "radioB",
    FeatureExtractorType.DINO_V2_VITS14.name: "dino",
    FeatureExtractorType.RGB.name: "rgb",
}
WORKFLOW_TYPE_TO_WANDB_PREFIX = {
    OsmoWorkflowType.EVAL.name: "eval",
    OsmoWorkflowType.DATAGEN.name: "gen",
    OsmoWorkflowType.TRAINING.name: "train",
    OsmoWorkflowType.TRAIN_AND_EVAL.name: "train_and_eval",
    OsmoWorkflowType.E2E.name: "e2e",
}

# FILE NAMES
TASK_TYPE_TO_HDF5_FILE_NAME = {
    Tasks.CUBE_STACKING.name: "PLACEHOLDER_HDF5_FILE_NAME_CUBES",
    Tasks.MUG_IN_DRAWER.name: "PLACEHOLDER_HDF5_FILE_NAME_MUG",
    Tasks.DRILL_IN_BOX.name: "PLACEHOLDER_HDF5_FILE_NAME_DRILL",
    Tasks.STICK_IN_BIN.name: "PLACEHOLDER_HDF5_FILE_NAME_STICK",
}
TASK_TYPE_TO_FPN_FILE_NAME = {
    Tasks.CUBE_STACKING.name: "PLACEHOLDER_FPN_FILE_NAME_CUBES",
    Tasks.MUG_IN_DRAWER.name: "PLACEHOLDER_FPN_FILE_NAME_MUG",
    Tasks.DRILL_IN_BOX.name: None,
    Tasks.STICK_IN_BIN.name: None,
}

# DEMO RANGES

# Data generation should at least cover the training+val set
TASK_TYPE_TO_DATAGEN_DEMO_RANGES = {
    Tasks.CUBE_STACKING.name: "0-149",
    Tasks.MUG_IN_DRAWER.name: "0-149",
    Tasks.DRILL_IN_BOX.name: "0-199",
    Tasks.STICK_IN_BIN.name: "0-199",
}

# Main training set
TASK_TYPE_TO_TRAINING_DEMO_RANGES = {
    Tasks.CUBE_STACKING.name: "0-129",
    Tasks.MUG_IN_DRAWER.name: "0-129",
    Tasks.DRILL_IN_BOX.name: "0-99",
    Tasks.STICK_IN_BIN.name: "0-99",
}

# Small range of demos used for validation during training
TASK_TYPE_TO_TRAINING_VALIDATION_DEMO_RANGES = {
    Tasks.CUBE_STACKING.name: "130-149",
    Tasks.MUG_IN_DRAWER.name: "130-149",
    Tasks.DRILL_IN_BOX.name: "100-119",
    Tasks.STICK_IN_BIN.name: "100-119",
}

# We evaluate on unseen data
TASK_TYPE_TO_EVALUATION_DEMO_RANGES = {
    Tasks.CUBE_STACKING.name: "150-249",
    Tasks.MUG_IN_DRAWER.name: "150-249",
    Tasks.DRILL_IN_BOX.name: "100-199",
    Tasks.STICK_IN_BIN.name: "100-199",
}


#### MISCELLANEOUS ####

# REQUIRED ARGS
REQUIRED_ARGS = {
    OsmoWorkflowType.EVAL.name: ["checkpoint", "feature_type", "task", "data_type"],
    OsmoWorkflowType.TRAINING.name: ["feature_type", "task", "data_type"],
    OsmoWorkflowType.DATAGEN.name: ["feature_type", "task", "data_type"],
    OsmoWorkflowType.TRAIN_AND_EVAL.name: ["feature_type", "task", "data_type"],
    OsmoWorkflowType.E2E.name: ["feature_type", "task", "data_type"],
}

# ARG CLASSES
TASK_TYPE_TO_ARG_CLS = {
    OsmoTaskType.TRAINING.name: TrainingAppArgs,
    OsmoTaskType.EVAL.name: ClosedLoopAppArgs,
    OsmoTaskType.DATAGEN.name: DataGenAppArgs,
}

# NODES
NNODES = 1

# TIMEOUTS
QUEUE_TIMEOUT = "48h"
EXEC_TIMEOUT = "336h"

# PATHS
mindmap_module_dir = os.path.dirname(mindmap_osmo.__file__)
WORKFLOW_FILE_PATH = os.path.join(mindmap_module_dir, "workflow_specs", "generated_workflow.yaml")

# URLS
SWIFT_URL = "PLACEHOLDER_SWIFT_URL"
SWIFT_BROWSEABLE_URL = "PLACEHOLDER_SWIFT_BROWSEABLE_URL"

# How often we're saving checkpoint to osmo during training
PERIODIC_CHECKPOINT_FREQUENCY = "15m"

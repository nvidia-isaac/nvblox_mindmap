# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from enum import Enum


class OsmoWorkflowType(Enum):
    """Workflow types supported by the OSMO workflow generation."""

    EVAL = "eval"
    DATAGEN = "datagen"
    TRAINING = "training"
    TRAIN_AND_EVAL = "train_and_eval"
    E2E = "e2e"


class OsmoTaskType(Enum):
    """OSMO task types supported by the OSMO workflow generation."""

    EVAL = "eval"
    DATAGEN = "datagen"
    TRAINING = "training"


class PlatformType(Enum):
    """Platform types supported by the OSMO workflow generation."""

    OVX_L40 = "ovx-l40"
    OVX_L40S = "ovx-l40s"
    DGX_H100 = "dgx-h100"


class DatasetType(Enum):
    """Osmo dataset types supported by the OSMO workflow generation."""

    CHECKPOINT = "checkpoint"  # OSMO dataset holding checkpoints.
    CHECKPOINT_SWIFT = "checkpoint_swift"  # Swiftstack directory with checkpoints
    HDF5 = "hdf5"  # OSMO dataset holding hdf5 files.
    DATA = "data"  # OSMO dataset holding the generated data.
    TASK = "task"  # Output of a task taking as input of a successor task.

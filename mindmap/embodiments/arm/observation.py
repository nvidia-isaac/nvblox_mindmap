# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from __future__ import annotations

from dataclasses import dataclass

from mindmap.embodiments.observation_base import ObservationBase
from mindmap.isaaclab_utils.isaaclab_camera_handler import IsaacLabCameraHandler

WRIST_RGB_ITEM_NAME = "wrist_rgb.png"
WRIST_DEPTH_ITEM_NAME = "wrist_depth.png"
WRIST_POSE_ITEM_NAME = "wrist_pose.npy"
WRIST_INTRINSICS_ITEM_NAME = "wrist_intrinsics.npy"
TABLE_RGB_ITEM_NAME = "table_rgb.png"
TABLE_DEPTH_ITEM_NAME = "table_depth.png"
TABLE_POSE_ITEM_NAME = "table_pose.npy"
TABLE_INTRINSICS_ITEM_NAME = "table_intrinsics.npy"


@dataclass
class ArmEmbodimentObservation(ObservationBase):
    table_camera: IsaacLabCameraHandler
    """External (table) camera"""

    wrist_camera: IsaacLabCameraHandler
    """Internal (wrist) camera"""


def get_camera_item_names_by_encoding_method(add_external_cam: bool):
    base_item_names = {
        "rgb": [
            WRIST_RGB_ITEM_NAME,
        ],
        "depth": [
            WRIST_DEPTH_ITEM_NAME,
            WRIST_POSE_ITEM_NAME,
            WRIST_INTRINSICS_ITEM_NAME,
        ],
    }
    if add_external_cam:
        base_item_names["rgb"].extend([TABLE_RGB_ITEM_NAME])
        base_item_names["depth"].extend(
            [
                TABLE_DEPTH_ITEM_NAME,
                TABLE_POSE_ITEM_NAME,
                TABLE_INTRINSICS_ITEM_NAME,
            ]
        )

    return base_item_names

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

POV_RGB_ITEM_NAME = "pov_rgb.png"
POV_DEPTH_ITEM_NAME = "pov_depth.png"
POV_POSE_ITEM_NAME = "pov_pose.npy"
POV_INTRINSICS_ITEM_NAME = "pov_intrinsics.npy"
EXTERNAL_RGB_ITEM_NAME = "external_rgb.png"
EXTERNAL_DEPTH_ITEM_NAME = "external_depth.png"
EXTERNAL_POSE_ITEM_NAME = "external_pose.npy"
EXTERNAL_INTRINSICS_ITEM_NAME = "external_intrinsics.npy"


@dataclass
class HumanoidEmbodimentObservation(ObservationBase):
    external_camera: IsaacLabCameraHandler
    """External camera"""

    pov_camera: IsaacLabCameraHandler
    """Head camera"""


def get_camera_item_names_by_encoding_method(add_external_cam: bool):
    base_item_names = {
        "rgb": [
            POV_RGB_ITEM_NAME,
        ],
        "depth": [
            POV_DEPTH_ITEM_NAME,
            POV_POSE_ITEM_NAME,
            POV_INTRINSICS_ITEM_NAME,
        ],
    }
    if add_external_cam:
        base_item_names["rgb"].extend([EXTERNAL_RGB_ITEM_NAME])
        base_item_names["depth"].extend(
            [
                EXTERNAL_DEPTH_ITEM_NAME,
                EXTERNAL_POSE_ITEM_NAME,
                EXTERNAL_INTRINSICS_ITEM_NAME,
            ]
        )

    return base_item_names

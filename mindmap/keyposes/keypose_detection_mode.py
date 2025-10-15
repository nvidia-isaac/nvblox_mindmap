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


class KeyposeDetectionMode(Enum):
    NONE = "none"  # No additional keypose detection (other than grasp moments)
    HIGHEST_Z_BETWEEN_GRASP = (
        "highest_z_between_grasp"  # Select the highest z point between two grasp events
    )
    HIGHEST_Z_OF_VERTICAL_MOTION = "highest_z_of_vertical_motion"  # For every vertical motion interval, select the highest z point
    HIGHEST_Z_OF_VERTICAL_MOTION_AND_HEAD_TURN = "highest_z_of_vertical_motion_and_head_turn"  # Additionally, add keyposes at head turn events


def has_highest_z_of_vertical_motion(keypose_detection_mode: KeyposeDetectionMode) -> bool:
    return keypose_detection_mode in [
        KeyposeDetectionMode.HIGHEST_Z_OF_VERTICAL_MOTION,
        KeyposeDetectionMode.HIGHEST_Z_OF_VERTICAL_MOTION_AND_HEAD_TURN,
    ]


def has_head_turn_events(keypose_detection_mode: KeyposeDetectionMode) -> bool:
    return keypose_detection_mode in [
        KeyposeDetectionMode.HIGHEST_Z_OF_VERTICAL_MOTION_AND_HEAD_TURN
    ]

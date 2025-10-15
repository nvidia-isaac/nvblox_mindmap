# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from mindmap.keyposes.keypose_detection_mode import KeyposeDetectionMode
from mindmap.tasks.tasks import Tasks

TASK_TYPE_TO_EXTRA_KEYPOSES_AROUND_GRASP_EVENTS = {
    Tasks.CUBE_STACKING.name: [5],
    Tasks.MUG_IN_DRAWER.name: [5, 15],
    Tasks.DRILL_IN_BOX.name: [5, 15],
    Tasks.STICK_IN_BIN.name: [5, 15],
}

TASK_TYPE_TO_KEYPOSE_DETECTION_MODE = {
    Tasks.CUBE_STACKING.name: KeyposeDetectionMode.HIGHEST_Z_BETWEEN_GRASP,
    Tasks.MUG_IN_DRAWER.name: KeyposeDetectionMode.HIGHEST_Z_OF_VERTICAL_MOTION,
    Tasks.DRILL_IN_BOX.name: KeyposeDetectionMode.HIGHEST_Z_OF_VERTICAL_MOTION_AND_HEAD_TURN,
    Tasks.STICK_IN_BIN.name: KeyposeDetectionMode.HIGHEST_Z_OF_VERTICAL_MOTION_AND_HEAD_TURN,
}

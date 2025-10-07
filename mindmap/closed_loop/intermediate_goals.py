# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from mindmap.tasks.tasks import Tasks

TASK_TYPE_TO_MAX_INTERMEDIATE_DISTANCE_M = {
    Tasks.CUBE_STACKING.name: None,
    Tasks.MUG_IN_DRAWER.name: None,
    Tasks.DRILL_IN_BOX.name: None,
    Tasks.STICK_IN_BIN.name: 0.3,
}

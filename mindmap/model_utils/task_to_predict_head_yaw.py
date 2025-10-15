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

TASK_TO_PREDICT_HEAD_YAW = {
    Tasks.CUBE_STACKING: False,
    Tasks.MUG_IN_DRAWER: False,
    Tasks.DRILL_IN_BOX: True,
    Tasks.STICK_IN_BIN: True,
}


def get_predict_head_yaw_from_task(task: Tasks | str) -> bool:
    """Get whether to predict head yaw from the task."""
    if isinstance(task, str):
        task = Tasks(task)
    return TASK_TO_PREDICT_HEAD_YAW[task]

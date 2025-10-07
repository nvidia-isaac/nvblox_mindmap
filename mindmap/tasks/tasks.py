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

CUBE_STACKING_TASK_NAME = "Isaac-Stack-Cube-Franka-With-Cams-IK-Rel-v0"
MUG_IN_DRAWER_TASK_NAME = "Isaac-Mug-in-Drawer-Franka-v0"
# NOTE(remos): In mindmap we do not distinguish between the left and right drill_in_box tasks.
DRILL_IN_BOX_RIGHT_TASK_NAME = "Isaac-Drill-In-Box-GR1T2-Right-v0"
DRILL_IN_BOX_LEFT_TASK_NAME = "Isaac-Drill-In-Box-GR1T2-Left-v0"
STICK_IN_BIN_RIGHT_TASK_NAME = "Isaac-Stick-In-Bin-GR1T2-Right-v0"
STICK_IN_BIN_LEFT_TASK_NAME = "Isaac-Stick-In-Bin-GR1T2-Left-v0"


class Tasks(Enum):
    """Tasks supported by the Diffuser Actor network."""

    CUBE_STACKING = "cube_stacking"
    MUG_IN_DRAWER = "mug_in_drawer"
    DRILL_IN_BOX = "drill_in_box"
    STICK_IN_BIN = "stick_in_bin"

    def to_full_task_name(self) -> str:
        """Convert task enum to full task name used by Isaac."""
        if self == Tasks.CUBE_STACKING:
            return CUBE_STACKING_TASK_NAME
        elif self == Tasks.MUG_IN_DRAWER:
            return MUG_IN_DRAWER_TASK_NAME
        elif self == Tasks.DRILL_IN_BOX:
            # Return the right task name by default
            return DRILL_IN_BOX_RIGHT_TASK_NAME
        elif self == Tasks.STICK_IN_BIN:
            # Return the right task name by default
            return STICK_IN_BIN_RIGHT_TASK_NAME
        else:
            raise ValueError(f"Unknown task: {self}")

    @staticmethod
    def from_full_task_name(task_name: str) -> "Tasks":
        """Convert full task name to task enum."""
        if task_name == CUBE_STACKING_TASK_NAME:
            return Tasks.CUBE_STACKING
        elif task_name == MUG_IN_DRAWER_TASK_NAME:
            return Tasks.MUG_IN_DRAWER
        elif task_name in [DRILL_IN_BOX_RIGHT_TASK_NAME, DRILL_IN_BOX_LEFT_TASK_NAME]:
            return Tasks.DRILL_IN_BOX
        elif task_name in [STICK_IN_BIN_RIGHT_TASK_NAME, STICK_IN_BIN_LEFT_TASK_NAME]:
            return Tasks.STICK_IN_BIN
        else:
            raise ValueError(f"Unknown task name: {task_name}")

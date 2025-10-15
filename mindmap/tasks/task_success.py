# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from isaaclab_tasks.manager_based.manipulation.stack import mdp as mdp_cube_stacking

from mindmap.tasks.task_definitions.drill_in_box.config.gr1 import mdp as mdp_drill_in_box
from mindmap.tasks.task_definitions.mug_in_drawer import mdp as mdp_mug_in_drawer
from mindmap.tasks.task_definitions.stick_in_bin.config.gr1 import mdp as mdp_stick_in_bin
from mindmap.tasks.tasks import Tasks


def get_task_outcome(task: Tasks, env) -> bool:
    if task == Tasks.CUBE_STACKING:
        return mdp_cube_stacking.cubes_stacked(env)
    elif task == Tasks.MUG_IN_DRAWER:
        return mdp_mug_in_drawer.object_in_drawer(env)
    elif task == Tasks.DRILL_IN_BOX:
        # NOTE(remos): Not checking hand height because it would be target_side dependent
        # and in mindmap we do not distinguish between the left and right drill_in_box tasks.
        return mdp_drill_in_box.object_in_box(env, check_hand_height=False)
    elif task == Tasks.STICK_IN_BIN:
        return mdp_stick_in_bin.object_in_drum(env, check_hand_height=False)
    else:
        raise ValueError("No outcome check for this task.")

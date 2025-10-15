# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
import torch

from mindmap.tasks.task_definitions.drill_in_box.config.gr1.mdp.target_side import TargetSide

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_in_drum(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    drum_cfg: SceneEntityCfg = SceneEntityCfg("open_drum"),
    check_hand_height: bool = True,
    min_hand_height_m: float = 0.25,
    target_side: str = TargetSide.UNDEFINED,
    drum_radius_m: float = 0.3,
    drum_height_m: float = 0.7,
    max_object_termination_vel_m_s: float | None = None,
) -> torch.Tensor:
    """Check if an object is dropped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    drum: RigidObject = env.scene[drum_cfg.name]

    # Get the object and drum bottom positions
    object_pos = object.data.root_pos_w
    object_vel = torch.abs(object.data.root_vel_w)
    drum_bottom_pos = drum.data.root_pos_w

    # Calculate horizontal distance from object to drum center
    horizontal_distance = torch.sqrt(
        (object_pos[:, 0] - drum_bottom_pos[:, 0]) ** 2
        + (object_pos[:, 1] - drum_bottom_pos[:, 1]) ** 2
    )

    # Check if object is within the circular area (x,y) and height bounds (z)
    object_in_circle = horizontal_distance <= drum_radius_m
    object_in_height_bounds = (
        object_pos[:, 2] > drum_bottom_pos[:, 2] - 1e-2
    ) & (  # 1 cm tolerance below
        object_pos[:, 2] < drum_bottom_pos[:, 2] + drum_height_m
    )  # Within drum height

    # Object is in drum if it's within the circle AND within height bounds
    object_in_drum = object_in_circle & object_in_height_bounds

    done = object_in_drum
    # Get the hand height relative to the environment origin
    if check_hand_height:
        if target_side == TargetSide.LEFT:
            hand_link_name = "left_hand_roll_link"
        elif target_side == TargetSide.RIGHT:
            hand_link_name = "right_hand_roll_link"
        else:
            raise ValueError(f"Invalid target side: {target_side}")
        robot_body_pos_w = robot.data.body_pos_w
        eef_idx = robot.data.body_names.index(hand_link_name)
        hand_height = robot_body_pos_w[:, eef_idx, 2] - env.scene.env_origins[:, 2]

        done = torch.logical_and(done, hand_height > min_hand_height_m)

    # Check that the object is not moving too fast
    if max_object_termination_vel_m_s is not None:
        done = torch.logical_and(done, object_vel[:, 0] < max_object_termination_vel_m_s)
        done = torch.logical_and(done, object_vel[:, 1] < max_object_termination_vel_m_s)
        done = torch.logical_and(done, object_vel[:, 2] < max_object_termination_vel_m_s)

    return done

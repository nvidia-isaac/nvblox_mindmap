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


def object_in_box(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("power_drill"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("open_box"),
    check_hand_height: bool = True,
    target_side: str = TargetSide.UNDEFINED,
    max_object_termination_vel_m_s: float | None = None,
) -> torch.Tensor:
    """Check if an object is dropped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]

    # Get the object and box bottom positions
    object_pos = object.data.root_pos_w
    object_vel = torch.abs(object.data.root_vel_w)
    box_bottom_pos = box.data.root_pos_w

    # Get the bounds wrt to the box bottom
    # TODO(remos): read size from env config
    box_size = torch.tensor([0.4, 0.3], device=env.device)
    BOX_HEIGHT = 0.2
    bounds_x_lower = box_bottom_pos[:, 0] - box_size[0] / 2
    bounds_x_upper = box_bottom_pos[:, 0] + box_size[0] / 2
    bounds_y_lower = box_bottom_pos[:, 1] - box_size[1] / 2
    bounds_y_upper = box_bottom_pos[:, 1] + box_size[1] / 2
    bounds_z_lower = box_bottom_pos[:, 2] - 1e-2  # 1 cm tolerance
    bounds_z_upper = box_bottom_pos[:, 2] + BOX_HEIGHT
    object_pos_in_bounds_x = (object_pos[:, 0] > bounds_x_lower) & (
        object_pos[:, 0] < bounds_x_upper
    )
    object_pos_in_bounds_y = (object_pos[:, 1] > bounds_y_lower) & (
        object_pos[:, 1] < bounds_y_upper
    )
    object_pos_in_bounds_z = (object_pos[:, 2] > bounds_z_lower) & (
        object_pos[:, 2] < bounds_z_upper
    )
    # Combine them to check if each environment’s object is fully in bounds
    object_in_box = object_pos_in_bounds_x & object_pos_in_bounds_y & object_pos_in_bounds_z

    done = object_in_box

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

        # Check that the right wrist is retracted back towards the body in z direction
        MIN_HAND_HEIGHT = 0.25
        done = torch.logical_and(done, hand_height > MIN_HAND_HEIGHT)

    # Check that the object is not moving too fast
    if max_object_termination_vel_m_s is not None:
        done = torch.logical_and(done, object_vel[:, 0] < max_object_termination_vel_m_s)
        done = torch.logical_and(done, object_vel[:, 1] < max_object_termination_vel_m_s)
        done = torch.logical_and(done, object_vel[:, 2] < max_object_termination_vel_m_s)

    return done


def object_too_close_to_robot(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("power_drill"),
    min_dist: float = 0.2,
) -> torch.Tensor:
    """Check if an object is too close to the robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    xy_dist = object.data.root_pos_w[..., 0:2] - robot.data.root_pos_w[..., 0:2]
    robot_too_close = torch.norm(xy_dist, dim=-1) < min_dist
    return robot_too_close

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import math
from typing import Tuple

import torch

from mindmap.geometry.pytorch3d_transforms import (
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_axis_angle,
)


def absolute_goal_from_relative(
    Eef_t_Eef_Goal: torch.Tensor,
    q_Eef_Goal: torch.Tensor,
    W_t_W_Eef: torch.Tensor,
    q_W_Eef: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes an absolute goal from a relative goal.

    The absolute goal is in the world-frame while the relative goal
    in the end-effector frame.

    Args:
        Eef_t_Eef_Goal (torch.Tensor): Translation from the goal frame to the end-effector frame,
            expressed in the end-effector frame.
        q_Eef_Goal (torch.Tensor): Rotation from the goal frame to the end-effector frame,
            expressed in the end-effector frame.
        W_t_W_Eef (torch.Tensor): Position of the end-effector in the world.
        q_W_Eef (torch.Tensor): Rotation of the end-effector in the world.


    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Position and Orientation of the goal in the world frame.
    """
    W_t_W_Goal = W_t_W_Eef + quaternion_apply(q_W_Eef, Eef_t_Eef_Goal)
    q_W_Goal = quaternion_multiply(q_W_Eef, q_Eef_Goal)
    return W_t_W_Goal, q_W_Goal


def get_error_to_goal(
    W_t_W_Eef: torch.Tensor,
    q_W_Eef: torch.Tensor,
    gripper_pos,
    W_t_W_Goal: torch.Tensor,
    q_W_Goal: torch.Tensor,
    gripper_closed_goal,
) -> Tuple[float, float]:
    """Compute the distance to a goal pose, in meters and degrees.

    Args:
        W_t_W_Eef (torch.Tensor): Position of the end-effector in the world.
        q_W_Eef (torch.Tensor): Rotation of the end-effector in the world.
        W_t_W_Goal (torch.Tensor): Position of the goal in the world.
        q_W_Goal (torch.Tensor): Rotation of the goal in the world.

    Returns:
        Tuple[float, float]: Position error (in meters), rotation error (in degrees).
    """
    # Translation error
    W_t_Eef_Goal = torch.norm(W_t_W_Eef - W_t_W_Goal)
    # Rotation error (in degrees)
    q_Eef_Goal = quaternion_multiply(quaternion_invert(q_W_Eef), q_W_Goal)
    aa_Eef_Goal = quaternion_to_axis_angle(q_Eef_Goal)
    angle_Eef_Goal_deg = math.degrees(torch.norm(aa_Eef_Goal))
    return W_t_Eef_Goal, angle_Eef_Goal_deg


def get_error_to_goal(
    W_t_W_Eef: torch.Tensor, q_W_Eef: torch.Tensor, W_t_W_Goal: torch.Tensor, q_W_Goal: torch.Tensor
) -> Tuple[float, float]:
    """Compute the distance to a goal pose, in meters and degrees.

    Args:
        W_t_W_Eef (torch.Tensor): Position of the end-effector in the world.
        q_W_Eef (torch.Tensor): Rotation of the end-effector in the world.
        W_t_W_Goal (torch.Tensor): Position of the goal in the world.
        q_W_Goal (torch.Tensor): Rotation of the goal in the world.

    Returns:
        Tuple[float, float]: Position error (in meters), rotation error (in degrees).
    """
    # Translation error
    W_t_Eef_Goal = torch.norm(W_t_W_Eef - W_t_W_Goal)
    # Rotation error (in degrees)
    q_Eef_Goal = quaternion_multiply(quaternion_invert(q_W_Eef), q_W_Goal)
    aa_Eef_Goal = quaternion_to_axis_angle(q_Eef_Goal)
    angle_Eef_Goal_deg = math.degrees(torch.norm(aa_Eef_Goal))
    return W_t_Eef_Goal, angle_Eef_Goal_deg

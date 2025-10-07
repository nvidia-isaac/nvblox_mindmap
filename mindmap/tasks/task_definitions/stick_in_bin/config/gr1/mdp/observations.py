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

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_left_eef_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_roll_link")
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins

    return left_eef_pos


def get_left_eef_quat(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_roll_link")
    left_eef_quat = body_quat_w[:, left_eef_idx]

    return left_eef_quat


def get_right_eef_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_roll_link")
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    return right_eef_pos


def get_right_eef_quat(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_roll_link")
    right_eef_quat = body_quat_w[:, right_eef_idx]

    return right_eef_quat


def get_hand_state(
    env: ManagerBasedRLEnv,
    num_hand_joints: int = 22,
) -> torch.Tensor:
    hand_joint_states = env.scene["robot"].data.joint_pos[:, -num_hand_joints:]

    return hand_joint_states


def get_head_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    robot_joint_names = env.scene["robot"].data.joint_names
    head_joint_names = ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]
    indexes = torch.tensor(
        [robot_joint_names.index(name) for name in head_joint_names], dtype=torch.long
    )
    head_joint_states = env.scene["robot"].data.joint_pos[:, indexes]

    return head_joint_states


def get_all_robot_link_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_link_state_w[:, :, :]
    all_robot_link_pos = body_pos_w

    return all_robot_link_pos

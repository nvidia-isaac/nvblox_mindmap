# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import torch

from mindmap.geometry.pytorch3d_transforms import quaternion_invert, quaternion_multiply


def get_current_pose_from_gripper_history(gripper_history: torch.Tensor) -> torch.Tensor:
    """
    Get the current pose from the gripper history.

    Args:
        gripper_history (torch.Tensor): The tensor representing the gripper history (batch_size, nhist, ngrippers, X).

    Returns:
        torch.Tensor: The current pose from the gripper history (batch_size, ngrippers, X).
    """
    # The current pose is the last gripper pose in the history.
    return gripper_history[:, -1, :, :]


def to_relative_pcd(pcd: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
    """
    Convert a point cloud to a relative coordinate system with respect to the current position.

    Args:
        pcd (torch.Tensor): The point cloud tensor.
        current_pose (torch.Tensor): The current pose tensor of shape (batch_size, X).

    Returns:
        torch.Tensor: The point cloud tensor in the relative coordinate system.
    """
    # NOTE(remos): Find out why we only translate and not rotate.
    current_position = current_pose[:, :3]  # (batch_size, 3)
    batch_size = current_position.shape[0]
    pcd = pcd - current_position.view(batch_size, 1, 3, 1, 1)
    return pcd


def to_relative_gripper_history(
    gripper_history: torch.Tensor, current_pose: torch.Tensor
) -> torch.Tensor:
    """
    Convert a gripper history relative to the current gripper position.

    Args:
        gripper_history (torch.Tensor): The gripper history tensor of shape (batch_size, nhist, ngrippers, X).
        current_pose (torch.Tensor): The current gripper pose tensor of shape (batch_size, ngrippers, X).

    Returns:
        torch.Tensor: The gripper history tensor in the relative coordinate system.
    """
    # NOTE(remos): Find out why we only translate and not rotate.
    current_position = current_pose[:, :, :3]  # (batch_size, ngrippers, 3)
    ngrippers = current_position.shape[1]
    batch_size = current_position.shape[0]
    gripper_history = gripper_history.clone()
    gripper_history[..., :3] = gripper_history[..., :3] - current_position.view(
        batch_size, 1, ngrippers, 3
    )
    return gripper_history


def to_relative_trajectory(trajectory: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
    """
    Convert a trajectory relative to the current gripper pose.

    Args:
        trajectory (torch.Tensor): The trajectory tensor of shape (batch_size, n, ngrippers, 8).
        current_pose (torch.Tensor): The current gripper pose tensor of shape (batch_size, ngrippers, X).

    Returns:
        torch.Tensor: The trajectory tensor in the relative coordinate system.
    """
    # NOTE(remos): Here we translate and rotate.
    # Check that trajectory consists out of translation + quaternion + gripper state.
    assert trajectory.shape[-1] == 8
    # Check that batch number is the same.
    assert trajectory.shape[0] == current_pose.shape[0]

    # Get position, quaternion and gripper state from the trajectory.
    absolute_position = trajectory[..., :3]
    absolute_quat = trajectory[..., 3:7]
    gripper_state = trajectory[..., 7].unsqueeze(-1)

    # Unsqueeze to add trajectory dimension.
    current_gripper_position = current_pose[..., :3].unsqueeze(1)
    current_gripper_quat = current_pose[..., 3:7].unsqueeze(1)

    # W_t_EE_P = W_t_W_P - W_t_W_EE
    relative_position = absolute_position - current_gripper_position
    # R_EE_P = inv(R_W_EE) * R_W_P
    relative_quat = quaternion_multiply(quaternion_invert(current_gripper_quat), absolute_quat)

    return torch.cat([relative_position, relative_quat, gripper_state], dim=-1)


def to_absolute_trajectory(trajectory: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
    """
    Convert a trajectory that is relative to the current gripper pose back to absolute.

    Args:
        trajectory (torch.Tensor): The trajectory tensor of shape (batch_size, n, 8).
        current_pose (torch.Tensor): The current gripper pose tensor of shape (batch_size, X).

    Returns:
        torch.Tensor: The trajectory tensor in the absolute coordinate system.
    """
    # NOTE(remos): Here we translate and rotate.
    # Check that trajectory consists out of translation + quaternion + gripper state.
    assert trajectory.shape[-1] == 8

    # Get position, quaternion and gripper state from the trajectory.
    relative_position = trajectory[..., :3]
    relative_quat = trajectory[..., 3:7]
    gripper_state = trajectory[..., 7].unsqueeze(-1)

    current_gripper_position = current_pose[..., :3].unsqueeze(1)
    current_gripper_quat = current_pose[..., 3:7].unsqueeze(1)

    # W_t_W_P = W_t_EE_P + W_t_W_EE
    absolute_pos = relative_position + current_gripper_position
    # R_W_P = R_W_EE * R_EE_P
    absolute_quat = quaternion_multiply(current_gripper_quat, relative_quat)

    return torch.cat([absolute_pos, absolute_quat, gripper_state], dim=-1)

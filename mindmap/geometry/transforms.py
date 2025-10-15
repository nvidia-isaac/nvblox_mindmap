# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import Tuple

import torch

from mindmap.geometry.pytorch3d_transforms import quaternion_to_matrix


def split_transformation_matrix(T_B_A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a 4x4 transformation matrix into a rotation matrix and a translation vector.

    Args:
        T_B_A (torch.Tensor): A 4x4 transformation matrix.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A 3x3 rotation matrix and 3x1 translation vector.
    """
    assert T_B_A.shape == torch.Size([4, 4])
    R_B_A = T_B_A[:3, :3]
    t_B_A = T_B_A[:3, 3]
    return R_B_A, t_B_A


def compose_transformation_matrix(R_B_A: torch.Tensor, t_B_A: torch.Tensor) -> torch.Tensor:
    """Compose a 3x3 rotation matrix and a 3x1 translation vector into a
       4x4 transformation matrix.

    Args:
        R_B_A (torch.Tensor): A 3x3 rotation matrix.
        t_B_A (torch.Tensor): A 3D translation vector

    Returns:
        torch.Tensor: A 4x4 transformation matrix.
    """
    assert R_B_A.shape == torch.Size([3, 3])
    assert t_B_A.numel() == 3
    return torch.vstack(
        (
            torch.hstack((R_B_A, t_B_A.reshape((3, 1)))),
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=t_B_A.device),
        )
    )


def transform(T_B_A: torch.Tensor, vec_A: torch.Tensor) -> torch.Tensor:
    """Applies a transformation matrix to a vector.

    Args:
        T_B_A (torch.Tensor): A 4x4 matrix transforming p_A to p_B.
        vec_A (torch.Tensor): A 3D vector in frame A.

    Returns:
        torch.Tensor: A 3D vector in frame B.
    """
    assert T_B_A.shape == torch.Size([4, 4])
    assert len(vec_A) == 3
    R_B_A, t_B_A = split_transformation_matrix(T_B_A)
    vec_B = torch.squeeze(torch.matmul(R_B_A, vec_A)) + t_B_A
    return vec_B


def look_at_to_rotation_matrix(
    center_W: torch.Tensor, look_at_point_W: torch.Tensor, camera_up_W: torch.Tensor
) -> torch.Tensor:
    """Generate a rotation matrix from a look-at view-point description.

    Args:
        center_W (torch.Tensor): The eye center in the world frame.
        look_at_point_W (torch.Tensor): The point the eye looks at in the world frame.
        camera_up_W (torch.Tensor): The up direction of the camera frame in the world frame.

    Returns:
        torch.Tensor: The 3x3 rotation matrix R_W_C rotating from camera to world.
    """
    assert len(center_W) == 3
    assert len(look_at_point_W) == 3
    assert len(camera_up_W) == 3
    # The camera-z is the unit vector pointing from the center to the look at point.
    z_vec = look_at_point_W - center_W
    z_vec = z_vec / torch.norm(z_vec)
    # Camera up is not necessarily perpendicular to z_vec, so use it to calculate x_vec
    x_vec = -1.0 * torch.cross(z_vec, camera_up_W)
    x_vec = x_vec / torch.norm(x_vec)
    # Calculate remaining vector
    y_vec = torch.cross(z_vec, x_vec)
    # Use the unit vectors to form the rotation matrix
    R_W_C = torch.hstack((x_vec.view((3, 1)), y_vec.view((3, 1)), z_vec.view((3, 1))))
    assert R_W_C.shape == torch.Size([3, 3])
    return R_W_C


def look_at_to_transformation_matrix(
    center_W: torch.Tensor, look_at_point_W: torch.Tensor, camera_up_W: torch.Tensor
) -> torch.Tensor:
    """Generate a transformation matrix from a look-at view-point description.

    Args:
        center_W (torch.Tensor): The eye center in the world frame.
        look_at_point_W (torch.Tensor): The point the eye looks at in the world frame.
        camera_up_W (torch.Tensor): The up direction of the camera frame in the world frame.

    Returns:
        torch.Tensor: The 4x4 transformation matrix R_W_C rotating from camera to world.
    """
    R_W_C = look_at_to_rotation_matrix(center_W, look_at_point_W, camera_up_W)
    t_W_C = center_W
    T_W_C = torch.vstack(
        (
            torch.hstack((R_W_C, t_W_C.view((3, 1)))),
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=center_W.device),
        )
    )
    assert T_W_C.shape == torch.Size([4, 4])
    return T_W_C


def transformation_trajectory_from_parts(
    eef_pos: torch.Tensor, eef_quat: torch.Tensor
) -> torch.Tensor:
    """Generates a transformation matrix trajectory from a position and quaternion trajectory.

    Args:
        eef_pos (torch.Tensor): Nx3 position trajectory.
        eef_quat (torch.Tensor): Nx4 quaternion trajectory.

    Returns:
        torch.Tensor: Nx4x4 transformation matrix trajectory.
    """
    assert eef_pos.shape[1] == 3
    assert eef_quat.shape[1] == 4
    eef_rot = quaternion_to_matrix(eef_quat)
    eef_pose = torch.zeros((eef_pos.shape[0], 4, 4), device=eef_pos.device)
    for idx in range(eef_pos.shape[0]):
        R_W_C = torch.squeeze(eef_rot[idx, :, :])
        t_W_C = torch.squeeze(eef_pos[idx, :])
        T_W_C = compose_transformation_matrix(R_W_C, t_W_C)
        eef_pose[idx, :, :] = T_W_C
    return eef_pose

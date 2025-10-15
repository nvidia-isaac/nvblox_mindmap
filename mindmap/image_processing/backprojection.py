# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import numpy as np
import numpy.typing as npt
import torch
from transforms3d.quaternions import quat2mat


def pose_to_homo(poses: torch.Tensor) -> torch.Tensor:
    """
    Convert a 7D pose representation (translation + quaternion) into a 4x4 homogeneous transformation matrix.

    Args:
        poses (torch.Tensor): A tensor of shape (..., 7) containing the poses, where the last dimension contains
            translation (x, y, z) followed by quaternion (qx, qy, qz, qw). Can be batched with arbitrary leading dimensions.

    Returns:
        torch.Tensor: A tensor of shape (..., 4, 4) containing homogeneous transformation matrices.
            The leading dimensions match the input pose tensor.
    """
    assert poses.ndim >= 1
    assert poses.shape[-1] == 7
    device = poses.device
    translation = poses[..., :3]

    # Handle batched quaternions
    quats = poses[..., 3:].cpu().numpy()
    if len(quats.shape) == 1:
        rotation_mat = torch.tensor(quat2mat(quats), device=device, dtype=torch.float32)
    else:
        # Apply quat2mat to each quaternion in the batch
        # TODO(remos): stay on the GPU here (replacing quat2mat)
        rotation_mats = np.stack([quat2mat(quat) for quat in quats])
        rotation_mat = torch.tensor(rotation_mats, device=device, dtype=torch.float32)

    # Create batched pose transform
    batch_size = poses.shape[0] if len(poses.shape) > 1 else 1
    pose_transform = torch.eye(4, device=device, dtype=torch.float32).repeat(batch_size, 1, 1)
    pose_transform[..., :3, :3] = rotation_mat
    pose_transform[..., :3, 3] = translation
    return pose_transform


def backproject_depth_to_pointcloud(
    depth_image: torch.Tensor, intrinsics: torch.Tensor, transform: torch.Tensor
) -> torch.Tensor:
    """
    Backproject a depth image to a 3D point cloud.

    Args:
        depth_image (torch.Tensor): Depth image tensor of shape (B, H, W)
        intrinsics (torch.Tensor): Camera intrinsic matrix tensor of shape (B, 3, 3)
        transform (torch.Tensor): Camera extrinsic matrix (pose) tensor of shape (B, 4, 4)

    Returns:
        torch.Tensor: Point cloud tensor in world coordinates of shape (B, H*W, 3)
    """
    assert depth_image.ndim == 3
    assert intrinsics.ndim == 3
    assert transform.ndim == 3
    assert depth_image.shape[0] == intrinsics.shape[0] == transform.shape[0]

    device = depth_image.device
    B, H, W = depth_image.shape

    # Create a meshgrid of image coordinates
    i, j = torch.meshgrid(
        torch.arange(W, device=device), torch.arange(H, device=device), indexing="xy"
    )

    # Expand for batch dimension and flatten spatial dims
    i_flat = i.expand(B, -1, -1).reshape(B, -1)
    j_flat = j.expand(B, -1, -1).reshape(B, -1)
    depth_flat = depth_image.reshape(B, -1)

    # Compute normalized coordinates for each batch
    uv1 = torch.stack((i_flat, j_flat, torch.ones_like(i_flat)), dim=2).to(
        torch.float32
    )  # (B, HW, 3)

    # Batch matrix multiply with intrinsics inverse
    # TODO(remos): check whether we need to add [0.5, 0.5] depending on how pixel centers are defined in IsaacLab
    K_inv = torch.linalg.inv(intrinsics)  # (B, 3, 3)
    xyz_camera = depth_flat.unsqueeze(-1) * (uv1 @ K_inv.transpose(1, 2))  # (B, HW, 3)

    # Homogeneous coordinates
    ones = torch.ones((*xyz_camera.shape[:-1], 1), device=xyz_camera.device)
    xyz_camera_homogeneous = torch.cat([xyz_camera, ones], dim=-1)  # (B, HW, 4)

    # Transform to world coordinates using extrinsic matrix
    xyz_world_homogeneous = xyz_camera_homogeneous @ transform.transpose(1, 2)  # (B, HW, 4)
    xyz_world = xyz_world_homogeneous[..., :3]  # (B, HW, 3)

    return xyz_world


def get_camera_pointcloud(
    intrinsics: torch.Tensor, depth: torch.Tensor, position: torch.Tensor, orientation: torch.Tensor
) -> torch.Tensor:
    """
    Generate a point cloud in world coordinates from the depth image.

    Args:
        intrinsics (torch.Tensor): Camera intrinsic matrix tensor of shape (3, 3) or (B, 3, 3)
        depth (torch.Tensor): Depth image tensor of shape (H, W) or (B, H, W)
        position (torch.Tensor): Camera position tensor of shape (3,) or (B, 3)
        orientation (torch.Tensor): Camera orientation quaternion tensor of shape (4,) or (B, 4)

    Returns:
        torch.Tensor: Point cloud data in world frame with shape (3, H, W) if input is unbatched,
                     or shape (B, 3, H, W) if input is batched. Points are represented as (x,y,z)
                     coordinates in the world frame. Invalid depth values are mapped to (0,0,0).
    """
    # Add a batch dimension if needed
    added_batch_dim = False
    if depth.ndim == 2:
        added_batch_dim = True
        intrinsics = intrinsics.unsqueeze(0)
        depth = depth.unsqueeze(0)
        position = position.unsqueeze(0)
        orientation = orientation.unsqueeze(0)
    assert intrinsics.ndim == 3
    assert depth.ndim == 3
    assert position.ndim == 2
    assert orientation.ndim == 2

    transform = pose_to_homo(torch.concatenate([position, orientation], axis=1))
    pointcloud = backproject_depth_to_pointcloud(depth, intrinsics, transform)
    pointcloud = torch.nan_to_num(pointcloud, nan=0.0, posinf=0.0, neginf=0.0)
    batch, height, width = depth.shape[:3]
    pointcloud = pointcloud.reshape([batch, height, width, 3])
    # Move channel dimension to front
    pointcloud = torch.permute(pointcloud, (0, 3, 1, 2))

    # Remove the batch dimension if we added it
    if added_batch_dim:
        pointcloud = pointcloud.squeeze(0)

    return pointcloud

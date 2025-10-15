# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import einops
import torch

from mindmap.geometry.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    matrix_to_quaternion,
    normalise_quat,
    quaternion_to_matrix,
)


def normalize_pointcloud(pcd: torch.Tensor, workspace_bounds: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a point cloud tensor by permuting the dimensions and applying a normalization (scaling) function.
    Args:
        pcd (torch.Tensor): The point cloud tensor to be normalized.
        workspace_bounds (torch.Tensor): The bounds of the workspace.
    Returns:
        torch.Tensor: The normalized point cloud tensor.
    """
    pcd_points_last = einops.rearrange(pcd, "b ncam dim3 w h -> b ncam w h dim3")
    pcd_points_last, valid_mask = normalize_pos(pcd_points_last, workspace_bounds)
    pcd = einops.rearrange(pcd_points_last, "b ncam w h dim3 -> b ncam dim3 w h")

    return pcd, valid_mask


def unnormalize_pointcloud(pcd: torch.Tensor, workspace_bounds: torch.Tensor) -> torch.Tensor:
    """
    Unnormalizes a point cloud tensor by permuting the dimensions and applying a normalization (scaling) function.
    Args:
        pcd (torch.Tensor): The point cloud tensor to be unnormalized.
        workspace_bounds (torch.Tensor): The bounds of the workspace.
    Returns:
        torch.Tensor: The unnormalized point cloud tensor.
    """
    pcd = pcd.clone()
    pcd = torch.permute(
        unnormalize_pos(torch.permute(pcd, [0, 1, 3, 4, 2]), workspace_bounds),
        [0, 1, 4, 2, 3],
    )
    return pcd


def normalize_trajectory(
    trajectory: torch.Tensor,
    workspace_bounds: torch.Tensor,
    rotation_parametrization,
    quaternion_format,
) -> torch.Tensor:
    """
    Normalizes a trajectory tensor applying a normalization (scaling) function to the position
    and converting the rotation to the given rotation parametrization.

    Args:
        trajectory (torch.Tensor): The trajectory tensor to be normalized.
        workspace_bounds (torch.Tensor): The bounds of the workspace.
        rotation_parametrization: The type of output rotation parametrization.
        quaternion_format: The format of the input quaternion.

    Returns:
        torch.Tensor: The normalized trajectory tensor.
    """
    assert trajectory.shape[-1] == 3 + 4
    trajectory = trajectory.clone()
    trajectory[..., :3], _ = normalize_pos(trajectory[..., :3], workspace_bounds)
    trajectory = convert_rot(trajectory, rotation_parametrization, quaternion_format)
    assert trajectory.shape[-1] == 9
    return trajectory


def unnormalize_trajectory(
    trajectory: torch.Tensor,
    workspace_bounds: torch.Tensor,
    rotation_parametrization,
    quaternion_format,
) -> torch.Tensor:
    """
    Unnormalizes a trajectory tensor by unnormalizing (scaling) the position
    and converting the rotation to the given quaternion format.

    Args:
        trajectory (torch.Tensor): The trajectory tensor to be unnormalized.
        workspace_bounds (torch.Tensor): The bounds of the workspace.
        rotation_parametrization: The type of input rotation parametrization.
        quaternion_format: The format of the output quaternion.

    Returns:
        torch.Tensor: The unnormalized trajectory tensor.
    """
    # Normalize quaternion
    if rotation_parametrization != "6D":
        trajectory[:, :, :, 3:7] = normalise_quat(trajectory[:, :, :, 3:7])
    # Back to quaternion
    trajectory = unconvert_rot(trajectory, rotation_parametrization, quaternion_format)
    # unnormalize position
    trajectory[:, :, :, :3] = unnormalize_pos(trajectory[:, :, :, :3], workspace_bounds)
    # Convert gripper status to probability
    if trajectory.shape[-1] > 7:
        trajectory[..., 7] = trajectory[..., 7].sigmoid()
    return trajectory


def convert_rot(signal: torch.Tensor, rotation_parametrization, quaternion_format) -> torch.Tensor:
    """
    Converts the rotation part of a signal tensor to a specified rotation parametrization.

    Args:
        signal (torch.Tensor): The input signal tensor.
        rotation_parametrization (str): The type of output rotation parametrization.
        quaternion_format (str): The format of the input quaternion.

    Returns:
        torch.Tensor: The converted signal tensor.
    """
    signal[..., 3:7] = normalise_quat(signal[..., 3:7])
    # BUG(xyao):
    #    assert rotation_parametrization in [
    # "quat_from_top_ghost", "quat_from_query",
    # "6D_from_top_ghost", "6D_from_query"
    # ]
    if "6D" in rotation_parametrization:
        # The following code expects wxyz quaternion format!
        if quaternion_format == "xyzw":
            signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
        rot = quaternion_to_matrix(signal[..., 3:7])

        # BUG (xyao): signal.size return ndim not shape
        res = signal[..., 7:] if signal.shape[-1] > 7 else None

        # we introduce a new dimension for the gripper in the code below
        if len(rot.shape) == 5:
            B, L, G, D1, D2 = rot.shape  # G is gripper dimension
            rot = rot.reshape(B * L * G, D1, D2)
            rot_6d = get_ortho6d_from_rotation_matrix(rot)
            rot_6d = rot_6d.reshape(B, L, G, 6)
        else:
            rot_6d = get_ortho6d_from_rotation_matrix(rot)
        signal = torch.cat([signal[..., :3], rot_6d], dim=-1)
        assert signal.shape[-1] == 3 + 6
        if res is not None:
            signal = torch.cat((signal, res), -1)
    return signal


def unconvert_rot(
    signal: torch.Tensor, rotation_parametrization, quaternion_format
) -> torch.Tensor:
    """
    Converts the rotation part of a signal tensor from a specified
    rotation parametrization to the given quaternion format.

    Args:
        signal (torch.Tensor): The input signal tensor.
        rotation_parametrization (str): The type of input rotation parametrization.
        quaternion_format (str): The format of the output quaternion.

    Returns:
        torch.Tensor: The converted signal tensor.
    """
    # BUG(xyao):
    #    assert rotation_parametrization in [
    # "quat_from_top_ghost", "quat_from_query",
    # "6D_from_top_ghost", "6D_from_query"
    # ]
    # BUG(xyao):
    # signal.shape(-1) not size(-1)
    if "6D" in rotation_parametrization:
        res = signal[..., 9:] if signal.shape[-1] > 9 else None
        if len(signal.shape) == 4:
            B, L, G, _ = signal.shape
            rot = signal[..., 3:9].reshape(B * L * G, 6)
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
            quat = quat.reshape(B, L, G, 4)
        else:
            rot = signal[..., 3:9]
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
        signal = torch.cat([signal[..., :3], quat], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        # The above code handled wxyz quaternion format!
        if quaternion_format == "xyzw":
            signal[..., 3:7] = signal[..., (4, 5, 6, 3)]
    return signal


def normalize_pos(pos: torch.Tensor, workspace_bounds: torch.Tensor) -> torch.Tensor:
    """
    Normalizes (scaling) the position tensor `pos` within the given `workspace_bounds` range.

    Args:
        pos (torch.Tensor): The position tensor to be normalized.
        workspace_bounds (torch.Tensor): The range of values within which the `pos` tensor should be normalized.

    Returns:
        torch.Tensor: The normalized position tensor.

    """
    pos = pos.clone()
    pos_min = workspace_bounds[0].float().to(pos.device)
    pos_max = workspace_bounds[1].float().to(pos.device)

    valid_mask = ((pos >= pos_min) & (pos <= pos_max)).all(dim=-1)

    return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0, valid_mask


def unnormalize_pos(pos: torch.Tensor, workspace_bounds: torch.Tensor) -> torch.Tensor:
    """
    Unnormalizes (scaling) the position tensor `pos` within the given `workspace_bounds` range.

    Args:
        pos (torch.Tensor): The position tensor to be unnormalized.
        workspace_bounds (torch.Tensor): The range of values within which the `pos` tensor should be unnormalized.

    Returns:
        torch.Tensor: The unnormalized position tensor.

    """
    pos_min = workspace_bounds[0].float().to(pos.device)
    pos_max = workspace_bounds[1].float().to(pos.device)
    return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

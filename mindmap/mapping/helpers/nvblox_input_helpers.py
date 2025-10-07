# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import Dict, List, Tuple

import torch

from mindmap.image_processing.backprojection import get_camera_pointcloud, pose_to_homo
from mindmap.isaaclab_utils.isaaclab_camera_handler import IsaacLabCameraHandler


def get_nvblox_inputs_from_sample(
    sample: Dict[str, torch.Tensor], camera_index: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the camera data formatted for nvblox integration from a sample dictionary.

    This method processes and converts various camera data (depth, RGB, pose, etc.) from a sample
    dictionary into the format required by the nvblox integrator.

    Args:
        sample (Dict[str, torch.Tensor]): Dictionary containing camera data with keys:
            - depths: Depth images tensor, shape (1, num_cams, H, W), dtype=float32
            - intrinsics: Camera intrinsic matrices, shape (1, num_cams, 3, 3), dtype=float32
            - camera_poses: Camera poses as quaternion+translation, shape (1, num_cams, 7), dtype=float32
            - rgbs: RGB images tensor, shape (1, num_cams, 3, H, W), dtype=float32, range [0,1]
            - segmentation_masks: Dynamic object masks, shape (1, num_cams, H, W), dtype=bool
        camera_index (int): Index of the camera to extract data for

    Returns:
        Tuple containing:
            depth_frame (torch.Tensor): Depth image, shape (H, W), dtype=float32
            intrinsics (torch.Tensor): Camera intrinsic matrix, shape (3, 3), dtype=float32
            camera_pose (torch.Tensor): Homogeneous transformation matrix, shape (4, 4), dtype=float32
            rgb (torch.Tensor): RGB image, shape (H, W, 3), dtype=uint8
            dynamic_mask (torch.Tensor): Boolean mask for dynamic objects, shape (H, W), dtype=bool
            pointcloud (torch.Tensor): World-frame pointcloud, shape (W, H, 3), dtype=float32
    """
    num_cams = sample["depths"].shape[1]
    assert camera_index < num_cams

    assert sample["depths"].shape == torch.Size([1, num_cams, 512, 512])
    assert sample["depths"].dtype == torch.float32
    depth_frame = sample["depths"].squeeze(0)[camera_index, ...]

    assert sample["intrinsics"].shape == torch.Size(
        [1, num_cams, 3, 3]
    ), f"intrinsics shape is {sample['intrinsics'].shape}"
    assert sample["intrinsics"].dtype == torch.float32
    intrinsics = sample["intrinsics"].squeeze(0)[camera_index, ...]

    assert sample["camera_poses"].shape == torch.Size(
        [1, num_cams, 7]
    ), f"camera_poses shape is {sample['camera_poses'].shape}"
    assert sample["camera_poses"].dtype == torch.float32
    camera_pose = sample["camera_poses"].squeeze(0)[camera_index, ...]
    camera_pose_homo = pose_to_homo(camera_pose).squeeze(0)

    assert sample["rgbs"].shape == torch.Size(
        [1, num_cams, 3, 512, 512]
    ), f"rgbs shape is {sample['rgbs'].shape}"
    assert torch.all(sample["rgbs"] >= 0) and torch.all(sample["rgbs"] <= 1)
    assert sample["rgbs"].dtype == torch.float32
    rgb = (sample["rgbs"].squeeze(0)[camera_index, ...].permute(1, 2, 0) * 255).to(torch.uint8)

    assert sample["segmentation_masks"].shape == torch.Size([1, num_cams, 512, 512])
    assert sample["segmentation_masks"].dtype == torch.bool
    dynamic_mask = sample["segmentation_masks"].squeeze(0)[camera_index]

    pointcloud = get_camera_pointcloud(
        intrinsics=intrinsics,
        depth=depth_frame,
        position=camera_pose[:3],
        orientation=camera_pose[3:],
    )

    return (depth_frame, intrinsics, camera_pose_homo, rgb, dynamic_mask, pointcloud)


def get_nvblox_inputs_from_camera_handler(
    camera_handler: IsaacLabCameraHandler, dynamic_class_labels: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the camera data formatted for nvblox integration.

    This method processes and converts various camera data (depth, RGB, pose, etc.) into the format
    required by the nvblox integrator. All tensors except intrinsics are moved to CUDA.

    Args:
        camera_handler (IsaacLabCameraHandler): The camera handler to get the data from.
        dynamic_class_labels (List[str]): List of semantic class labels to use for the dynamic mask.

    Returns:
        Tuple containing:
            depth_frame (torch.Tensor): Depth image, shape (H, W), dtype=float32
            intrinsics (torch.Tensor): Camera intrinsic matrix, shape (3, 3), dtype=float32
            camera_pose (torch.Tensor): Homogeneous transformation matrix, shape (4, 4), dtype=float32
            rgb (torch.Tensor): RGB image, shape (H, W, 3), dtype=uint8
            dynamic_mask (torch.Tensor): Boolean mask for dynamic objects, shape (H, W), dtype=bool
            pointcloud (torch.Tensor): World-frame pointcloud, shape (W, H, 3), dtype=float32
    """
    dynamic_mask = (
        camera_handler.get_dynamic_segmentation(dynamic_class_labels).squeeze().to("cuda")
    )
    assert dynamic_mask.dtype == torch.bool

    depth_frame = camera_handler.get_depth().to("cuda")
    assert depth_frame.dtype == torch.float32

    intrinsics = camera_handler.get_intrinsics().to("cuda")
    assert intrinsics.dtype == torch.float32

    camera_pose = camera_handler.get_pose_as_homo().to(torch.float32).to("cuda")

    rgb = camera_handler.get_rgb().to(torch.uint8).to("cuda")

    pointcloud = camera_handler.get_pcd()
    assert pointcloud.dtype == torch.float32

    return (depth_frame, intrinsics, camera_pose, rgb, dynamic_mask, pointcloud)

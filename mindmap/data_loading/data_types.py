# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from enum import Enum

#### DATA TYPE ####


class DataType(Enum):
    RGBD = "rgbd"
    MESH = "mesh"
    RGBD_AND_MESH = "rgbd_and_mesh"


def includes_rgb(data_type: DataType) -> bool:
    """Check if the data type includes RGB images."""
    rgb_methods = {DataType.RGBD, DataType.RGBD_AND_MESH}
    return data_type in rgb_methods


def includes_depth_camera(data_type: DataType) -> bool:
    """Check if the data type includes depth camera data."""
    depth_camera_methods = {DataType.RGBD, DataType.RGBD_AND_MESH}
    return data_type in depth_camera_methods


def includes_pcd(data_type: DataType) -> bool:
    """Check if the data type converts depth images to pointclouds."""
    pcd_methods = {DataType.RGBD, DataType.RGBD_AND_MESH}
    return data_type in pcd_methods


def includes_mesh(data_type: DataType) -> bool:
    """Check if the data type includes mesh data."""
    mesh_methods = {DataType.MESH, DataType.RGBD_AND_MESH}
    return data_type in mesh_methods


def includes_policy_states(data_type: DataType) -> bool:
    """Check if the data type includes gripper state information."""
    gripper_states_methods = {
        DataType.RGBD,
        DataType.MESH,
        DataType.RGBD_AND_MESH,
    }
    return data_type in gripper_states_methods


def includes_nvblox(data_type: DataType) -> bool:
    """Check if the data type includes nvblox data structures."""
    nvblox_methods = {DataType.RGBD_AND_MESH, DataType.MESH}
    return data_type in nvblox_methods

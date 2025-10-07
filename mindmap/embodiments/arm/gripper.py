# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import Optional

import numpy.typing as npt
import torch

from mindmap.embodiments.arm.robot_state import ArmEmbodimentRobotState

# This is the threshold to use for the gripper jaw positions to determine if
# the gripper is open or closed.
GRIPPER_OPEN_THRESHOLD = 0.04 - 1e-4


def is_gripper_closed(gripper_pos: torch.Tensor) -> torch.Tensor:
    """Function returning whether the gripper is closed based on the gripper positions.

    Args:
        gripper_1_pos (torch.Tensor): Tensor of size N indicating the position of gripper 1.
        gripper_2_pos (torch.Tensor): Tensor of size N indicating the position of gripper 2.

    Returns:
        torch.Tensor (Size 1): Whether the gripper is closed.
                               We assume it is closed as soon as it is not fully open.
    """
    # We use the positions of the gripper teeth to infer whether the gripper is closed.
    # The gripper position is 0.04 if fully open and smaller otherwise.
    # We assume the gripper is closed as soon as it is not fully open.
    # If the gripper pos is just one value we unsqueeze it.
    if gripper_pos.dim() == 1:
        gripper_pos = gripper_pos.unsqueeze(0)
    gripper_1_pos = gripper_pos[:, 0]
    gripper_2_pos = gripper_pos[:, 1]
    assert gripper_1_pos.dim() == 1
    assert gripper_2_pos.dim() == 1
    assert len(gripper_1_pos) == len(gripper_2_pos)
    num_measurements = len(gripper_1_pos)
    gripper_close_binary = torch.logical_and(
        gripper_1_pos < GRIPPER_OPEN_THRESHOLD, gripper_2_pos < GRIPPER_OPEN_THRESHOLD
    ).reshape([num_measurements])
    assert gripper_close_binary.dim() == 1
    assert len(gripper_close_binary) == num_measurements
    return gripper_close_binary


def is_gripper_open_numpy(gripper_pos: npt.NDArray) -> bool:
    # This function if a convenient numpy interface to the torch version
    # that we're using in data generation pipeline.
    assert gripper_pos.ndim == 1
    assert gripper_pos.shape[0] == 2
    # Call out to the torch version.
    return torch.logical_not(is_gripper_closed(torch.tensor(gripper_pos))).item()

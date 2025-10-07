# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import os

from importlib_resources import files
import numpy as np
import torch

from mindmap.model_utils.relative_conversions import (
    get_current_pose_from_gripper_history,
    to_absolute_trajectory,
    to_relative_trajectory,
)
from mindmap.tests.utils.comparisons import quaternion_is_close
from mindmap.tests.utils.constants import TestDataLocations

HISTORY_LENGTH = 3
PREDICTION_HORIZON = 5
NUM_GRIPPERS = 1
TRAJECTORY_POSE_LENGTH = 3 + 4 + 1  # translation + rotation + gripper state

GRIPPER_HISTORY_FILE = os.path.join(TestDataLocations.test_data_dir, "0003.gripper_history.npy")
GT_GRIPPER_PRED_FILE = os.path.join(TestDataLocations.test_data_dir, "0003.gt_gripper_pred.npy")


def test_inverse_conversion():
    """Test that the relative conversion is the inverse of the absolute conversion."""
    gripper_history = torch.tensor(np.load(GRIPPER_HISTORY_FILE))
    gt_gripper_pred = torch.tensor(np.load(GT_GRIPPER_PRED_FILE))

    # Add batch dimension.
    gripper_history = gripper_history.unsqueeze(0)
    gt_gripper_pred = gt_gripper_pred.unsqueeze(0)

    # Get current pose from gripper history.
    current_pose = get_current_pose_from_gripper_history(gripper_history)

    # Check input dimensions
    assert gripper_history.shape == (1, HISTORY_LENGTH, NUM_GRIPPERS, TRAJECTORY_POSE_LENGTH)
    assert current_pose.shape == (1, NUM_GRIPPERS, TRAJECTORY_POSE_LENGTH)
    assert gt_gripper_pred.shape == (1, PREDICTION_HORIZON, NUM_GRIPPERS, TRAJECTORY_POSE_LENGTH)

    # Convert to relative and back to absolute.
    relative_trajectory = to_relative_trajectory(gt_gripper_pred, current_pose)
    absolute_trajectory = to_absolute_trajectory(relative_trajectory, current_pose)

    # Check that after the inverse conversion, the trajectories are the same.
    assert absolute_trajectory.shape == gt_gripper_pred.shape
    assert torch.isclose(
        absolute_trajectory[..., :3], gt_gripper_pred[..., :3], rtol=1e-3, atol=1e-5
    ).all()
    assert quaternion_is_close(absolute_trajectory[..., 3:7], gt_gripper_pred[..., 3:7])
    assert torch.isclose(
        absolute_trajectory[..., 7], gt_gripper_pred[..., 7], rtol=1e-3, atol=1e-5
    ).all()

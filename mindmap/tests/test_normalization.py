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

import numpy as np
import torch

from mindmap.model_utils.normalization import (
    normalize_pointcloud,
    normalize_trajectory,
    unnormalize_pointcloud,
    unnormalize_trajectory,
)
from mindmap.tests.utils.comparisons import quaternion_is_close
from mindmap.tests.utils.constants import TestDataLocations

PREDICTION_HORIZON = 5
NUM_GRIPPERS = 1
TRAJECTORY_POSE_LENGTH = 3 + 4 + 1  # translation + rotation + gripper state

TRAJECTORY_FILE = os.path.join(TestDataLocations.test_data_dir, "0003.gt_gripper_pred.npy")

rotation_parametrization: str = "6D_from_query"
quaternion_format: str = "wxyz"
workspace_bounds = torch.tensor([[0.378, -0.143, -0.033], [0.606, 0.232, 0.369]])


def test_inverse_normalization():
    """Test if normalization and unnormalization are inverse operations."""
    trajectory = torch.tensor(np.load(TRAJECTORY_FILE))

    # Add batch dimension.
    trajectory = trajectory.unsqueeze(0)

    # Check input dimensions
    assert trajectory.shape == (1, PREDICTION_HORIZON, NUM_GRIPPERS, TRAJECTORY_POSE_LENGTH)
    trajectory = trajectory[..., :7]

    # Convert to relative and back to absolute.
    normalized_trajectory = normalize_trajectory(
        trajectory, workspace_bounds, rotation_parametrization, quaternion_format
    )
    unnormalized_trajectory = unnormalize_trajectory(
        normalized_trajectory, workspace_bounds, rotation_parametrization, quaternion_format
    )

    # Check that after the inverse conversion, the trajectories are the same.
    assert unnormalized_trajectory.shape == trajectory.shape
    assert torch.isclose(
        unnormalized_trajectory[..., :3], trajectory[..., :3], rtol=1e-3, atol=1e-5
    ).all()
    assert quaternion_is_close(unnormalized_trajectory[..., 3:7], trajectory[..., 3:7])


def test_pointcloud_normalization():
    # Create a random pointcloud
    pointcloud = torch.randn(1, 1, 3, 100, 100)

    # Create some random (but valid) gripper loc bounds
    workspace_bounds = torch.rand(2, 3)
    workspace_bounds[0] = workspace_bounds[0] * -1.0

    normalized_pointcloud, valid_mask = normalize_pointcloud(pointcloud, workspace_bounds)
    # Check that the pointcloud and normalized pointcloud are different
    assert torch.max(torch.abs(normalized_pointcloud - pointcloud)) > 1e-3

    unnormalized_pointcloud = unnormalize_pointcloud(normalized_pointcloud, workspace_bounds)

    # Check that the pointcloud and unnormalized pointcloud are the same, if we remove the points out of bounds
    mask_expand = valid_mask.expand(1, 1, 3, 100, 100)
    assert (
        torch.max(torch.abs(unnormalized_pointcloud[mask_expand] - pointcloud[mask_expand])) < 1e-3
    )

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import Callable, List, Tuple

import numpy as np
import torch


def get_grasp_events(
    gripper_pos: torch.Tensor,
    gripper_speed_threshold: float,
    is_gripper_open: Callable[[torch.Tensor], bool],
    smoothing_kernel_size: int = 2,
    plot: bool = False,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Detect grasping event intervals and gripper openess

    Args:
        gripper_pos: torch.Tensor of shape (N, M)
            - N: is the history length
            - M: the number of gripper parts (teeth, fingers, etc.)
        gripper_speed_threshold: float
            - The minimum speed of the gripper to be considered as moving.
        smoothing_kernel_size: int
            - The size of the smoothing kernel to use.
        is_gripper_open: Callable[[torch.Tensor], bool]
            - A function that returns True if the gripper is open, False otherwise
            - input is a torch.Tensor of shape (M,).
              Where M is the number of gripper parts (teeth, fingers, etc.)
    Returns:
        grasp_intervals: List of 2-tuples defining start/end indices of each grasp event
        gripper_open: np.ndarray of shape (N,). 1 is open, 0 is closed.
    """
    # Compute speed of gripper
    # Convert list of tensors to list of numpy arrays
    gripper_pos = np.array([pos.numpy() for pos in gripper_pos])
    gripper_pos_norm = np.linalg.norm(gripper_pos, axis=1)
    gripper_speed = np.abs(np.diff(gripper_pos_norm, n=1))  # First-order diff
    gripper_speed[0] = gripper_speed[-1] = 0  # Handle edge points

    # Apply smoothing. This has the effect of
    # * Making sure there's motion detected throughout a gripping interval.
    # * Extending the interval in both directions.
    smoothing_kernel = np.ones(smoothing_kernel_size) / smoothing_kernel_size
    gripper_speed = np.convolve(gripper_speed, smoothing_kernel)

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(gripper_speed)
        plt.title("Smoothed Gripper Speed")
        plt.xlabel("Time")
        plt.ylabel("Speed")
        plt.show()

    # Find mask indicating non-zero speed.
    pos_change_mask = gripper_speed > gripper_speed_threshold

    # Now find out where the mask is changing to determine start/end of gripper intervals.
    # mask_diff will contain +1 for start of gripper interval, and -1 for end of interval
    mask_diff = np.diff(pos_change_mask, prepend=0, append=0)
    start_indices = np.where(mask_diff == 1)[0]
    end_indices = np.where(mask_diff == -1)[0]

    # If the sequence ended before the gripper was closed, we might be missing an end index. This
    # should normally not happen.
    if len(end_indices) < len(start_indices):
        end_indices.append(len(gripper_pos) - 1)
    assert len(end_indices) == len(start_indices)

    grasp_intervals = list(zip(start_indices, end_indices))
    # Now figure out where the gripper is open and closed
    gripper_open = np.zeros(len(gripper_pos))

    # Find gripper state at start of sequence (should normally be open)
    current_gripper_open = is_gripper_open(gripper_pos[0, :])
    prev_end_index = 0

    for interval in grasp_intervals:
        # If gripper is open, we set it to to closed at the *last* frame in the grasping event. This
        # is the point in time where we start to hold on to the object. -1 to make sure the gripper
        # isn't fully closed
        if current_gripper_open:
            next_end_index = max(interval[1] - 1, 0)
        # If gripper is closed, we set it to open at the *first* frame in the grasping event. This
        # is the point in tme where we let go of the opbject. +1 to make sure the gripper has started to move.
        else:
            next_end_index = min(interval[0] + 1, len(gripper_open))

        gripper_open[prev_end_index:next_end_index] = current_gripper_open

        prev_end_index = next_end_index
        current_gripper_open = not current_gripper_open

    # Don't forget to set the state of the remainig indices, after the last grasping event
    gripper_open[prev_end_index:] = current_gripper_open
    return grasp_intervals, gripper_open

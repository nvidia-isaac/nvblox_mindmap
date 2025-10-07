# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import numpy as np
import numpy.typing as npt
import torch


class KeyposeOnlineEstimatorBase(ABC):
    """Base class for keypose estimators.

    Keypose estimators extract keypose indices from a sequence of robot states.

    """

    def extract_keypose_indices(
        self,
        gripper_states,
        extra_keyposes_around_grasp_events,
        keypose_detection_mode,
        use_keyposes,
    ):
        """Extract keypose events from a sequence of robot states."""
        raise NotImplementedError


def ensure_first_and_last_frames_are_keyposes(
    keypose_indices: npt.NDArray, num_frames: int
) -> npt.NDArray:
    """Ensure the first and last frames are keyposes.

    Args:
        keypose_indices: np.array of keypose indices
        num_frames: int of the number of frames in the sequence

    Returns:
        keypose_indices: np.array of keypose indices
    """
    keypose_list = keypose_indices.tolist()
    if len(keypose_indices) == 0 or (not keypose_indices[-1] == (num_frames - 1)):
        keypose_list.append(num_frames - 1)

    if len(keypose_indices) == 0 or (not keypose_indices[0] == 0):
        keypose_list.insert(0, 0)

    return np.array(keypose_list)


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


def get_extra_keypose_indices_around_intervals(
    grasp_intervals: List[Tuple[int, int]],
    extra_keyposes_around_grasp_events: List[int],
    length: int,
) -> List[int]:
    """Generate a list of keypose indices from grasp intervals.

    Args:
        grasp_intervals: List of 2-tuples defining start/end indices of each grasp event
        extra_keyposes_around_grasp_events: List of integers defining the number of extra keypose indices to add around each grasp event
        length: The length of the sequence of robot states

    Returns:
        extra_keypose_indices: List of integers defining the keypose indices
    """
    extra_keypose_indices = []
    for index in extra_keyposes_around_grasp_events:
        for interval in grasp_intervals:
            before_idx = interval[0] - index
            after_idx = interval[1] + index
            if before_idx >= 0:
                extra_keypose_indices.append(before_idx)
            if after_idx < length:
                extra_keypose_indices.append(after_idx)
    return extra_keypose_indices


def get_highest_z_of_vertical_motion(
    grasp_intervals: List[Tuple[int, int]],
    eef_pos: List[torch.Tensor],
    window_size: int = 5,
    min_vertical_motion_ratio: float = 0.6,
    min_vertical_motion_interval_length: int = 2,
    min_between_grasp_interval: int = 50,
    min_vertical_diff_m: float = 0.05,
) -> List[int]:
    """Get the highest z-value of vertical motion intervals

    Args:
    grasp_intervals: List of 2-tuples defining start/end indices of each grasp event
    eef_pos: torch.Tensor of shape (N, 3)
    window_size: Window size for the moving average on the vertical motion ratio to detect vertical motion segments.
    min_vertical_motion_ratio: Minimum ratio of the vertical motion to the total motion in vertical motion segments.
    min_vertical_motion_interval_length: Minimum number of frames in a vertical motion segment.
    min_between_grasp_interval: Minimum number of frames between two grasping events to detect vertical motion segments.
    min_vertical_diff_m: Minimum vertical difference in meters in a vertical motion segment.

    Returns:
    filtered_vertical_indices: List of indices of the highest z-value of vertical motion intervals
    vertical_motion_mask: Mask indicating for which input frames the gripper is open.
    """
    # Convert list of tensors to a numpy array
    eef_pos = np.array([pos.numpy() for pos in eef_pos])

    # For every pose in the trajectory, compute the velocity vector and the vertical motion ratio.
    velocities = np.diff(eef_pos, axis=0)
    velocities_norm = np.linalg.norm(velocities, axis=1)

    # Get around division by zero.
    velocities_norm[velocities_norm <= 1e-6] = 1e-6
    velocities_normalized = velocities / velocities_norm[:, np.newaxis]
    vertical_motion_ratio = np.abs(velocities_normalized[:, 2])

    # Smooth the vertical motion ratio with a moving average.
    smoothed_vertical_motion_ratio = np.zeros_like(vertical_motion_ratio)
    for i in range(len(vertical_motion_ratio)):
        start_idx = max(0, i - window_size)
        end_idx = min(len(vertical_motion_ratio), i + window_size + 1)
        smoothed_vertical_motion_ratio[i] = np.mean(vertical_motion_ratio[start_idx:end_idx])

    # Generate a vertical motion mask indicating where the vertical motion ratio is above a threshold.
    vertical_motion_mask = smoothed_vertical_motion_ratio > min_vertical_motion_ratio

    # Check for vertical motion changes.
    # If a direction change is detected, we split the vertical motion segment by
    # marking the index at which the change occurs as non-vertical motion.
    for i in range(1, len(vertical_motion_mask) - 1):
        if vertical_motion_mask[i]:
            previous_z_diff = eef_pos[i][2] - eef_pos[i - 1][2]
            next_z_diff = eef_pos[i + 1][2] - eef_pos[i][2]
            if previous_z_diff * next_z_diff < 0:
                vertical_motion_mask[i] = False

    # Find continuous segments of vertical motion from the vertical motion mask.
    start_idx = None
    vertical_motion_segments = []
    for i in range(len(vertical_motion_mask)):
        if vertical_motion_mask[i] and start_idx is None:
            # Start of vertical motion
            start_idx = i
        elif not vertical_motion_mask[i] and start_idx is not None:
            # End of vertical motion
            if i - start_idx > min_vertical_motion_interval_length:
                vertical_motion_segments.append((start_idx, i))
            start_idx = None
    if start_idx is not None:
        vertical_motion_segments.append((start_idx, len(vertical_motion_mask)))

    # If there are no grasping events, just return the vertical motion mask.
    if len(grasp_intervals) == 0:
        return [], vertical_motion_mask

    # Find vertical motion segment between grasping events and
    # add the highest pose of each vertical motion segment as a keypose:
    # If there are multiple vertical motion segments in the same grasping interval, only add one upward and one downward motion keypose.
    # Select the last downward motion keypose and the first upward motion keypose in a grasping interval.
    filtered_vertical_indices = []
    for grasp_interval_idx in range(-1, len(grasp_intervals)):
        # Find the indices of the end of the last grasp and the start of the next grasp interval.
        # These two indices define the interval between two grasping events in which we search for vertical motion segments.
        if grasp_interval_idx == -1:
            end_last_grasp_idx = 0
        else:
            end_last_grasp_idx = grasp_intervals[grasp_interval_idx][1]
        if grasp_interval_idx == len(grasp_intervals) - 1:
            start_next_grasp_idx = len(eef_pos)
        else:
            start_next_grasp_idx = grasp_intervals[grasp_interval_idx + 1][0]
        # Skip if the interval between two grasping events is too short.
        if start_next_grasp_idx - end_last_grasp_idx < min_between_grasp_interval:
            print(
                f"Skipping vertical motion segment. Interval between two grasping events \
                  {min_between_grasp_interval} is too short: {start_next_grasp_idx} - {end_last_grasp_idx}"
            )
            continue

        # For each interval between two grasping events, search if any vertical motion segment is present.
        upward_motions = []
        downward_motions = []
        for vertical_motion_segment in vertical_motion_segments:
            vertical_motion_start_idx = vertical_motion_segment[0]
            vertical_motion_end_idx = vertical_motion_segment[1]
            z_diff = abs(
                eef_pos[vertical_motion_end_idx][2] - eef_pos[vertical_motion_start_idx][2]
            )
            if min_vertical_diff_m is not None and z_diff < min_vertical_diff_m:
                continue
            # Check if the vertical motion is upward or downward.
            if eef_pos[vertical_motion_end_idx][2] > eef_pos[vertical_motion_start_idx][2]:
                # Check if the upward motion ends between the last grasp and the next grasp.
                if (
                    vertical_motion_end_idx >= end_last_grasp_idx
                    and vertical_motion_end_idx < start_next_grasp_idx
                ):
                    # We add the last pose of the upward motion as we always want to select the highest pose of the vertical motion segment.
                    upward_motions.append(vertical_motion_end_idx)
            else:
                # Check if the downward motion starts between the last grasp and the next grasp.
                if (
                    vertical_motion_start_idx >= end_last_grasp_idx
                    and vertical_motion_start_idx < start_next_grasp_idx
                ):
                    # We add the first pose of the downward motion as we always want to select the highest pose of the vertical motion segment.
                    downward_motions.append(vertical_motion_start_idx)

        if len(upward_motions) > 0:
            # If there are multiple upward motions between the two grasping events only use the first one.
            filtered_vertical_indices.append(upward_motions[0])
        if len(downward_motions) > 0:
            # If there are multiple downward motions between the two grasping events only use the last one.
            filtered_vertical_indices.append(downward_motions[-1])

    return filtered_vertical_indices, vertical_motion_mask


def select_indices_between_grasps(
    indices: List[int], grasp_intervals: List[Tuple[int, int]]
) -> List[int]:
    """
    Select indices that fall between the end of the first grasp interval and the start of the last grasp interval.

    Args:
        indices: List of integer indices to filter.
        grasp_intervals: List of 2-tuples (start_idx, end_idx) defining grasp intervals.

    Returns:
        indices_between_grasps: List of indices that are greater than the end of the first grasp interval
                                and less than the start of the last grasp interval.
    """
    indices_between_grasps = []
    for idx in indices:
        if idx > grasp_intervals[0][1] and idx < grasp_intervals[-1][0]:
            indices_between_grasps.append(idx)
    return indices_between_grasps


def get_extra_keyposes_between_indices(
    indices: List[int], min_interval_distance: int, fractions: List[float]
) -> List[int]:
    """
    Generate extra keypose indices at specified fractional positions between all pairs of sorted indices,
    provided the interval between them exceeds a minimum distance.

    Args:
        indices: List of integer indices.
        min_interval_distance: Minimum distance between two indices to consider adding extra keyposes.
        fractions: List of floats in [0, 1] specifying the relative positions within the interval
                   at which to add extra keypose indices.

    Returns:
        extra_keypose_indices: List of integer indices at the specified fractions between interval pairs.
    """
    extra_keypose_indices = []
    sorted_indices = sorted(indices)
    for interval_idx in range(0, len(sorted_indices) - 1, 2):
        last_interval_end_idx = sorted_indices[interval_idx]
        next_interval_start_idx = sorted_indices[interval_idx + 1]
        interval_distance = next_interval_start_idx - last_interval_end_idx
        if interval_distance > min_interval_distance:
            for fraction in fractions:
                assert fraction > 0 and fraction < 1, "Fraction must be between 0 and 1"
                idx_at_fraction = int(last_interval_end_idx + fraction * interval_distance)
                extra_keypose_indices.append(idx_at_fraction)
    return extra_keypose_indices


def get_previous_keypose(keypose_indices: List[int], current_idx: int) -> int:
    """
    Find the index in keypose_indices that is closest before current_idx.

    Args:
        keypose_indices: List of integer indices representing keyposes.
        current_idx: The current index to search before.

    Returns:
        The largest keypose index less than current_idx. If none found, returns 0.
    """
    prev_indices = sorted([i for i in keypose_indices if i < current_idx])
    if len(prev_indices) > 0:
        return prev_indices[-1]
    else:
        # The first frame is always a keypose
        return 0


def intervals_to_indices(intervals: List[Tuple[int, int]]) -> List[int]:
    """Convert a list of intervals to a list of indices.

    Args:
        intervals: List of 2-tuples defining start/end indices of each interval

    Returns:
        indices: List of integers defining the indices
    """
    if len(intervals) == 0:
        return []
    return np.concatenate(intervals)


def combine_indices(*args: List[int]) -> List[int]:
    """Combine several index lists into a single list of unique indices.

    Args:
        *args: List of lists of integers

    Returns:
        indices: List of integers defining the indices
    """
    indices = np.concatenate(args).astype(np.int32)
    return np.unique(np.sort(indices))

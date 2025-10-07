# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from __future__ import annotations

from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks
import torch

from mindmap.embodiments.arm.gripper import is_gripper_open_numpy
from mindmap.embodiments.arm.robot_state import ArmEmbodimentRobotState
from mindmap.embodiments.keypose_estimation_base import (
    KeyposeOnlineEstimatorBase,
    combine_indices,
    ensure_first_and_last_frames_are_keyposes,
    get_extra_keypose_indices_around_intervals,
    get_grasp_events,
    get_highest_z_of_vertical_motion,
    intervals_to_indices,
)
from mindmap.keyposes.keypose_detection_mode import KeyposeDetectionMode


class ArmEmbodimentKeyposeEstimator(KeyposeOnlineEstimatorBase):
    def __init__(self):
        self.gripper_speed_threshold = 0.0025
        self.gripper_open_threshold = 0.0350

    def _get_highest_z_between_grasps(
        self, grasp_intervals: List[Tuple[int, int]], eef_pos: List[torch.Tensor]
    ) -> List[int]:
        """Get the highest z-value between grasping events

        Args:
        grasp_intervals: List of 2-tuples defining start/end indices of each grasp event
        eef_pos: torch.Tensor of shape (N, 3)

        Returns:
        maxz_indices: List of indices of the highest z-value between grasping events
        """
        # Convert list of tensors to a numpy array
        eef_pos = np.array([pos.numpy() for pos in eef_pos])
        maxz_indices = []
        for i in range(len(grasp_intervals) - 1):
            idx = grasp_intervals[i][1]  # grasp-event end index
            next_idx = grasp_intervals[i + 1][0]  # Start index of next grasp-event

            # don't extract new keypose close to current ones
            margin = 2

            local_z_values = eef_pos[idx + margin : next_idx - margin][:, 2]
            local_peak_indices = find_peaks(local_z_values)[0]
            if len(local_peak_indices) > 0:
                largest_peak_index = (
                    margin
                    + idx
                    + local_peak_indices[np.argsort(local_z_values[local_peak_indices])[-1]]
                )
                maxz_indices.append(largest_peak_index)
        return maxz_indices

    def extract_keypose_indices(
        self,
        gripper_states: List[ArmEmbodimentRobotState],
        extra_keyposes_around_grasp_events: List[int],
        keypose_detection_mode: KeyposeDetectionMode,
    ) -> Tuple[npt.NDArray[np.int32], List[ArmEmbodimentRobotState]]:
        """Extract keyposes indices and gripper_open mask

        A keypose will be detected when:
        * Begin/end of a grasping event (non-zero gripper speed)
        * Depending on keypose_detection_mode:
            * Maxima in Z (height) between grasping events
            * Maxima in Z (height) of vertical motion intervals
        * First and last frames are always keyposes

        In addition, additional keyposes can be inserted around grasping events by providing indices in
        extra_keyposes_around_grasp_events. For example, giving [1,3,5] will create keyposes at the
        first, third and fifth frames before and after a grasping event. The rational here is to enable
        finer motion prediction in sections where the gripper operates close to the target object and
        might potentiall knock it over.

        The gripper state is changing to *closed* whenever we are fully grasping an object, i.e as soon
        as the closing grippers stop moving.

        The gripper state is changing to *open* as soon as soon as we are not fully grasping an object,
        i.e. as soon as the closed grippers starts to move.

        Args:
        gripper_pos Nx2 array of xy gripper positions.
        eef_pos     Nx3 array of end effector positions.
        extra_keyposes_around_grasp_events List of extra keypose indices around grasp events.
        keypose_detection_mode: KeyposeDetectionMode

        Returns:

        keypose_indices: List of extracted keyposes.

        """
        # Corner case
        if len(gripper_states) == 1:
            return [0]

        # Get the eef positions
        eef_pos = [state.W_t_W_Eef for state in gripper_states]

        # Get the grasp intervals
        grasp_intervals, _ = self.get_grasp_events(gripper_states)

        if keypose_detection_mode == KeyposeDetectionMode.HIGHEST_Z_BETWEEN_GRASP:
            maxz_indices = self._get_highest_z_between_grasps(grasp_intervals, eef_pos)
        elif keypose_detection_mode == KeyposeDetectionMode.HIGHEST_Z_OF_VERTICAL_MOTION:
            maxz_indices, _ = get_highest_z_of_vertical_motion(
                grasp_intervals,
                eef_pos,
                # Disabling the minimal vertical diff filter as it was added for the drill_in_box task
                # and was not needed to get the mug in drawer task going.
                # TODO(remos): Consider enabling the filter and testing it on the mug in drawer task.
                min_vertical_diff_m=None,
            )
        else:
            raise NotImplementedError(
                f"Keypose detection mode not implemented: {keypose_detection_mode}"
            )

        # Add some extra keyposes close to the grasping events
        extra_keypose_indices = get_extra_keypose_indices_around_intervals(
            grasp_intervals, extra_keyposes_around_grasp_events, len(gripper_states)
        )

        # Combine all the indices
        keypose_indices = combine_indices(
            intervals_to_indices(grasp_intervals),
            maxz_indices,
            extra_keypose_indices,
        )

        # Ensure the first and last frames are keyposes
        keypose_indices = ensure_first_and_last_frames_are_keyposes(
            keypose_indices, len(gripper_states)
        )

        return keypose_indices

    def plot_3d_keyposes(keypose_indices, eef_pos, gripper_open_mask, vertical_motion_mask=None):
        """Create a 3d plot of keyposes"""
        # Plot the full trajectory and keyposes in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the full trajectory with color coding for vertical motion
        for i in range(len(eef_pos) - 1):
            if vertical_motion_mask is None:
                ax.plot(
                    eef_pos[i : i + 2, 0],
                    eef_pos[i : i + 2, 1],
                    eef_pos[i : i + 2, 2],
                    color="gray",
                    alpha=0.5,
                )
            elif vertical_motion_mask[i]:
                ax.plot(
                    eef_pos[i : i + 2, 0],
                    eef_pos[i : i + 2, 1],
                    eef_pos[i : i + 2, 2],
                    color="orange",
                    alpha=0.8,
                )
            else:
                ax.plot(
                    eef_pos[i : i + 2, 0],
                    eef_pos[i : i + 2, 1],
                    eef_pos[i : i + 2, 2],
                    color="gray",
                    alpha=0.5,
                )

        # Highlight the keyposes with different colors based on gripper state
        valid_indices = (keypose_indices >= 0) & (keypose_indices < len(gripper_open_mask))
        valid_keypose_indices = keypose_indices[valid_indices]
        open_keyposes = valid_keypose_indices[gripper_open_mask[valid_keypose_indices]]
        closed_keyposes = valid_keypose_indices[~gripper_open_mask[valid_keypose_indices]]

        # Plot open gripper keyposes in purple
        ax.scatter(
            eef_pos[open_keyposes, 0],
            eef_pos[open_keyposes, 1],
            eef_pos[open_keyposes, 2],
            color="purple",
            label="Open Gripper Keyposes",
        )

        # Plot closed gripper keyposes in red
        ax.scatter(
            eef_pos[closed_keyposes, 0],
            eef_pos[closed_keyposes, 1],
            eef_pos[closed_keyposes, 2],
            color="red",
            label="Closed Gripper Keyposes",
        )

        # Mark the initial position with 'o'
        ax.scatter(
            eef_pos[0, 0],
            eef_pos[0, 1],
            eef_pos[0, 2],
            color="green",
            marker="o",
            s=100,
            label="Start Position",
        )

        # Mark the end position with 'x'
        ax.scatter(
            eef_pos[-1, 0],
            eef_pos[-1, 1],
            eef_pos[-1, 2],
            color="blue",
            marker="x",
            s=100,
            label="End Position",
        )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("3D Trajectory with Keyposes")
        ax.legend()

        plt.show()

    def get_grasp_events(
        self, robot_states: List[ArmEmbodimentRobotState]
    ) -> Tuple[List[Tuple[int, int]], npt.NDArray]:
        # Get the gripper jaw positions
        gripper_jaw_positions = [state.gripper_jaw_positions for state in robot_states]
        # Find indices that indicate grasp events + mask of openess indicator
        grasp_intervals, gripper_open_mask = get_grasp_events(
            gripper_pos=gripper_jaw_positions,
            gripper_speed_threshold=self.gripper_speed_threshold,
            is_gripper_open=is_gripper_open_numpy,
        )
        return grasp_intervals, gripper_open_mask

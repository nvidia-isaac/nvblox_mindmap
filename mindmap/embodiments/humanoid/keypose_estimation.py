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

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from mindmap.embodiments.humanoid.hand import (
    CLOSED_THRESHOLD,
    get_tensor_of_proximal_joints,
    is_hand_closed_instantaneous_from_proximal_joint_states,
)
from mindmap.embodiments.humanoid.robot_state import HumanoidEmbodimentRobotState
from mindmap.embodiments.keypose_estimation_base import (
    KeyposeOnlineEstimatorBase,
    combine_indices,
    ensure_first_and_last_frames_are_keyposes,
    get_extra_keypose_indices_around_intervals,
    get_extra_keyposes_between_indices,
    get_highest_z_of_vertical_motion,
    get_previous_keypose,
    intervals_to_indices,
    select_indices_between_grasps,
)
from mindmap.keyposes.keypose_detection_mode import (
    KeyposeDetectionMode,
    has_head_turn_events,
    has_highest_z_of_vertical_motion,
)

CLOSE_INTERVAL_THRESHOLD_DEFAULT = 10


@dataclass
class HumanoidHandProximalJointStates:
    """A dataclass containing the proximal joint states and their names for each hand."""

    # The proximal joint states
    # - A tensor of shape (num_samples, num_proximal_joints) containing the joint
    #   states of the proximal joints of a hand. The proximal joints are the joints
    #   are a subset of the hand joints.
    left_hand_proximal_joint_states: torch.Tensor
    right_hand_proximal_joint_states: torch.Tensor

    # The proximal joint names
    # - A list of strings, each containing the full name of each proximal joint
    #   in the tensor above.
    left_hand_proximal_joint_names: List[str]
    right_hand_proximal_joint_names: List[str]


@dataclass
class HumanoidGraspIntervals:
    """A dataclass containing the grasp intervals and hand open masks for each hand."""

    # Grasp intervals
    # - A list of tuples, each containing the start and end indices of a grasp event.
    left_hand_grasp_intervals: List[Tuple[int, int]]
    right_hand_grasp_intervals: List[Tuple[int, int]]

    # Hand open masks
    # - A tensor of shape (num_samples, ) containing a boolean value for each sample
    #   indicating whether the hand is open.
    left_hand_open_masks: torch.Tensor
    right_hand_open_masks: torch.Tensor


class HumanoidEmbodimentKeyposeEstimator(KeyposeOnlineEstimatorBase):
    def __init__(self):
        # Detection parameters
        # The threshold below which we consider the hand to be closed, when in open state (hysteresis)
        self.closed_threshold = CLOSED_THRESHOLD
        # The threshold above which we consider the hand to be open, when in closed state (hysteresis)
        self.open_threshold = -0.2
        # This is the joint speed to at which we consider the "gripper" (hand)
        # to be moving for grasp detection.
        self.velocity_threshold = 0.01
        # This is the size of the smoothing kernel to use for smoothing the
        # hand joint velocity for grasp detection.
        self.smoothing_kernel_size = 2

    def extract_keypose_indices(
        self,
        robot_states: List[HumanoidEmbodimentRobotState],
        extra_keyposes_around_grasp_events: List[int],
        keypose_detection_mode: KeyposeDetectionMode,
        plot=False,
    ) -> npt.NDArray[np.int32]:
        """Detect keyposes from a robot state sequence.

        This function *currently* detects keyposes based on:

        - Grasp intervals: Beginning and end of a grasping event (non-zero gripper speed)
        - Extra keyposes: Keyposes at specified intervals before and after grasping events.
        - TODO: We need to add more conditions likely. But these are not currently added.

        Args:
            robot_states: List of N RobotStateBase objects to a N x num_joints tensor of joint states per hand
            extra_keyposes_around_grasp_events: List of extra keypose indices around grasp events.
            keypose_detection_mode: KeyposeDetectionMode
            plot: Whether to plot the results.

        Returns:
            keypose_indices: List of keypose indices.
        """
        # Get the grasp intervals and relevant joint states
        grasp_intervals = self.get_grasp_events(robot_states, plot=plot)

        assert len(grasp_intervals.left_hand_open_masks) == len(
            grasp_intervals.right_hand_open_masks
        )
        assert len(grasp_intervals.left_hand_open_masks) == len(robot_states)
        left_hand_positions = [state.W_t_W_LeftEef for state in robot_states]
        right_hand_positions = [state.W_t_W_RightEef for state in robot_states]

        # Detect keyposes for each hand.
        keypose_indices = []
        for hand_grasp_intervals, eef_positions in zip(
            [
                grasp_intervals.left_hand_grasp_intervals,
                grasp_intervals.right_hand_grasp_intervals,
            ],
            [
                left_hand_positions,
                right_hand_positions,
            ],
        ):
            vertical_motion_keypose_indices = []
            extra_vertical_motion_keypose_indices = []
            if has_highest_z_of_vertical_motion(keypose_detection_mode):
                # Get keyposes at vertical motion intervals.
                vertical_motion_keypose_indices, _ = get_highest_z_of_vertical_motion(
                    hand_grasp_intervals,
                    eef_positions,
                )
                # Only allow vertical motion keyposes between grasps
                # (i.e. not before the first grasp or after the last grasp).
                vertical_motion_keypose_indices = select_indices_between_grasps(
                    vertical_motion_keypose_indices, hand_grasp_intervals
                )
                # Get extra keyposes between vertical motion keyposes.
                extra_vertical_motion_keypose_indices = get_extra_keyposes_between_indices(
                    indices=vertical_motion_keypose_indices,
                    min_interval_distance=10,
                    fractions=[0.5],
                )
            elif keypose_detection_mode == KeyposeDetectionMode.HIGHEST_Z_BETWEEN_GRASP:
                raise NotImplementedError(
                    f"{keypose_detection_mode} not implemented for humanoid embodiment."
                )
            elif keypose_detection_mode == KeyposeDetectionMode.NONE:
                print(
                    f"Selected {keypose_detection_mode} for humanoid embodiment. "
                    "Only using grasp and head turn events."
                )
            else:
                raise NotImplementedError(
                    f"Keypose detection mode not implemented: {keypose_detection_mode}"
                )

            # Grasp events are keyposes.
            hand_grasp_keypose_indices = intervals_to_indices(hand_grasp_intervals)
            # Add extra keyposes around grasp events.
            extra_grasp_keypose_indices = get_extra_keypose_indices_around_intervals(
                hand_grasp_intervals,
                extra_keyposes_around_grasp_events=extra_keyposes_around_grasp_events,
                length=len(robot_states),
            )

            # Add all keyposes of this hand.
            keypose_indices = combine_indices(
                keypose_indices,
                hand_grasp_keypose_indices,
                extra_grasp_keypose_indices,
                vertical_motion_keypose_indices,
                extra_vertical_motion_keypose_indices,
            )

        # TODO(alexmillane: 2025.05.28): As in the arm keypose estimation, we will likely need
        # extra keyposes at other times pre/post grasp intervals.
        # For the time being I'm just going to see how things go with the grasp events only.

        # Adding keypose indices for head turn events
        if has_head_turn_events(keypose_detection_mode):
            head_turn_events = self.get_head_turn_events(robot_states, keypose_indices.tolist())
            keypose_indices = combine_indices(keypose_indices, head_turn_events)

        # Ensure the first and last frames are keyposes
        keypose_indices = ensure_first_and_last_frames_are_keyposes(
            keypose_indices, len(robot_states)
        )

        return np.array(keypose_indices, dtype=np.int32)

    def get_grasp_events(
        self,
        robot_states: List[HumanoidEmbodimentRobotState],
        plot: bool = False,
    ) -> HumanoidGraspIntervals:
        # List of N RobotStateBase objects to a N x num_joints tensor of joint states per hand
        left_hand_joint_states = torch.stack([g.left_hand_joint_states for g in robot_states])
        right_hand_joint_states = torch.stack([g.right_hand_joint_states for g in robot_states])

        left_hand_grasp_intervals, left_hand_gripper_open = self.get_grasp_events_from_single_hand(
            left_hand_joint_states, plot=plot, hand_name="Left"
        )
        (
            right_hand_grasp_intervals,
            right_hand_gripper_open,
        ) = self.get_grasp_events_from_single_hand(
            right_hand_joint_states, plot=plot, hand_name="Right"
        )

        grasp_intervals = HumanoidGraspIntervals(
            left_hand_grasp_intervals=left_hand_grasp_intervals,
            right_hand_grasp_intervals=right_hand_grasp_intervals,
            left_hand_open_masks=torch.from_numpy(left_hand_gripper_open).to("cuda"),
            right_hand_open_masks=torch.from_numpy(right_hand_gripper_open).to("cuda"),
        )
        return grasp_intervals

    def get_head_turn_events(
        self,
        robot_states: List[HumanoidEmbodimentRobotState],
        keypose_indices: List[int],
        min_yaw_diff_rad: float = 45.0 * torch.pi / 180.0,
    ) -> List[int]:
        """
        Detects key head turn events from a sequence of robot states.

        This method examines the sequence of head yaw angles and detects indices where the direction
        of head rotation changes (i.e., where the sign of the yaw derivative changes). To reduce
        noise from minor oscillations, it only considers events where the head yaw differs by at least
        `min_yaw_diff_rad` (default: 45 degrees) from the previous turn event or keypose.

        Args:
            robot_states (List[HumanoidEmbodimentRobotState]): Sequence of robot states containing head yaw angles.
            keypose_indices (List[int]): List of existing keypose indices to consider for filtering.
            min_yaw_diff_rad (float, optional): Minimum required yaw difference (in radians) between consecutive key events.

        Returns:
            List[int]: Indices corresponding to detected key head turn events.
        """
        # Compute head yaw differences in radians
        head_yaw_rad = torch.stack([g.head_yaw_rad for g in robot_states])
        yaw_diffs = torch.diff(head_yaw_rad, dim=0)

        # Find indices where yaw diffs change direction (sign change)
        sign_change_mask = (yaw_diffs[:-1] * yaw_diffs[1:]) < 0  # True where sign changes
        # Indices where sign change occurs (between i and i+1, so assign to i+1)
        sign_change_indices = torch.where(sign_change_mask)[0] + 1

        # Now filter so that each keypose is at least 10 deg away from the last
        head_turn_indices = []
        for idx in sign_change_indices:
            # Check that the last keypose is at least min_yaw_diff_rad away from the current index.
            previous_keypose = get_previous_keypose(head_turn_indices + keypose_indices, idx)
            previous_yaw = head_yaw_rad[previous_keypose]
            if abs(head_yaw_rad[idx] - previous_yaw) > min_yaw_diff_rad:
                head_turn_indices.append(int(idx))

        return head_turn_indices

    def get_grasp_events_from_single_hand(
        self, hand_joint_states: torch.Tensor, plot: bool = False, hand_name: Optional[str] = None
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        # Check we've been passed a tensor of single hand joint states
        assert hand_joint_states.ndim == 2
        assert hand_joint_states.shape[1] == HumanoidEmbodimentRobotState.num_joints_per_hand()

        # Make things a little more readable
        class ClosednessState(Enum):
            OPEN = 0
            CLOSED = 1

        # Get the proximal joint states
        proximal_joint_states, proximal_joint_names = get_tensor_of_proximal_joints(
            hand_joint_states
        )

        # Initial state
        start_closedness = (
            ClosednessState.CLOSED
            if is_hand_closed_instantaneous_from_proximal_joint_states(proximal_joint_states[0, :])
            else ClosednessState.OPEN
        )

        # Hysteresis-based state transitions
        closedness_states = []
        transition_states = []
        transition_indices = []
        closedness_state = start_closedness
        for idx in range(proximal_joint_states.shape[0]):
            if closedness_state == ClosednessState.OPEN:
                fingers_closed = torch.any(proximal_joint_states[idx, :] < self.closed_threshold)
                if fingers_closed:
                    closedness_state = ClosednessState.CLOSED
                    transition_states.append(closedness_state)
                    transition_indices.append(idx)
            else:
                fingers_closed = torch.all(proximal_joint_states[idx, :] > self.open_threshold)
                if fingers_closed:
                    closedness_state = ClosednessState.OPEN
                    transition_states.append(closedness_state)
                    transition_indices.append(idx)
            closedness_states.append(closedness_state.value)

        # Calculate velocities
        proximal_joint_velocities = torch.abs(torch.diff(proximal_joint_states, dim=0))
        smoothing_kernel = np.ones(self.smoothing_kernel_size) / self.smoothing_kernel_size
        smoothed_joint_velocities = []
        for i in range(proximal_joint_velocities.shape[1]):
            smoothed_joint_velocities.append(
                torch.from_numpy(
                    np.convolve(proximal_joint_velocities[:, i].cpu().numpy(), smoothing_kernel)
                )
            )
        proximal_joint_smoothed_joint_velocities = torch.stack(smoothed_joint_velocities, dim=-1)

        # For each transition index, work backwards until the velocity is below a threshold
        start_indices = []
        for transition_idx in transition_indices:
            # Work backwards
            while transition_idx > 0:
                transition_idx -= 1
                if torch.any(
                    proximal_joint_smoothed_joint_velocities[transition_idx, :]
                    < self.velocity_threshold
                ):
                    break
            start_indices.append(transition_idx)
        end_indices = transition_indices
        assert len(start_indices) == len(end_indices)
        grasp_intervals = list(zip(start_indices, end_indices))
        assert len(closedness_states) == proximal_joint_states.shape[0]
        gripper_open = (~np.array(closedness_states).astype(bool)).astype(int)

        # Filter grasp intervals that are too close together.
        # We assume these are quick open/close events generated by noisy teleop data.
        demo_length = hand_joint_states.shape[0]
        grasp_intervals = self.filter_close_intervals(grasp_intervals, demo_length)

        # Plot if requested
        if plot:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(3, 1, 1)
            ax.plot(proximal_joint_states.cpu().numpy(), label=proximal_joint_names)
            for start_idx, end_idx in grasp_intervals:
                plt.axvline(x=start_idx, color="g", linestyle="--", alpha=0.5)
                plt.axvline(x=end_idx, color="r", linestyle="--", alpha=0.5)
            plt.axhline(y=self.closed_threshold, color="k", linestyle="--", alpha=0.5)
            plt.axhline(y=self.open_threshold, color="k", linestyle="--", alpha=0.5)
            plt.grid()
            plt.title("Proximal Joints")
            plt.legend()
            ax = fig.add_subplot(3, 1, 2)
            ax.plot(gripper_open, "b", label="left hand")
            plt.grid()
            plt.title("Openness State")
            ax = fig.add_subplot(3, 1, 3)
            plt.plot(proximal_joint_velocities.cpu().numpy(), "r", label="left hand")
            plt.plot(proximal_joint_smoothed_joint_velocities.cpu().numpy(), "b", label="left hand")
            for start_idx, end_idx in grasp_intervals:
                plt.axvline(x=start_idx, color="g", linestyle="--", alpha=0.5)
                plt.axvline(x=end_idx, color="r", linestyle="--", alpha=0.5)
            plt.title("Velocity")
            plt.grid()
            if hand_name is not None:
                plt.suptitle(f"{hand_name} Hand")
            plt.show()
        return grasp_intervals, gripper_open

    def filter_close_intervals(
        self, grasp_intervals: List[Tuple[int, int]], demo_length: int
    ) -> List[Tuple[int, int]]:
        """
        Filters out grasp intervals that are too close to each other, or too close to the start or end of the demonstration.
        This is used to remove spurious or noisy open/close events in teleoperation data.

        Args:
            grasp_intervals (List[Tuple[int, int]]): List of (start_idx, end_idx) tuples representing detected grasp intervals.
            demo_length (int): The total number of timesteps in the demonstration.

        Returns:
            List[Tuple[int, int]]: Filtered list of grasp intervals, with close or noisy intervals removed.
        """
        filtered_intervals = []
        for idx_current, current_interval in enumerate(grasp_intervals):
            found_close_interval = False
            for idx_test, test_interval in enumerate(grasp_intervals):
                if idx_current == idx_test:
                    # Don't compare interval to itself
                    continue
                if (
                    self.are_close_intervals(
                        current_interval,
                        test_interval,
                    )
                    or self.interval_close_to_demo_start(
                        current_interval,
                    )
                    or self.interval_close_to_demo_end(
                        current_interval,
                        demo_length,
                    )
                ):
                    found_close_interval = True
            if not found_close_interval:
                # We only keep the intervals that are not close to any other interval
                # or the start or end of the demo
                filtered_intervals.append(current_interval)
        return filtered_intervals

    @staticmethod
    def are_close_intervals(
        grasp_interval_1: Tuple[int, int],
        grasp_interval_2: Tuple[int, int],
        close_interval_threshold: int = CLOSE_INTERVAL_THRESHOLD_DEFAULT,
    ) -> bool:
        """
        Determines whether two grasp intervals are close to each other in time.

        Args:
            grasp_interval_1 (Tuple[int, int]): The first grasp interval as (start_idx, end_idx).
            grasp_interval_2 (Tuple[int, int]): The second grasp interval as (start_idx, end_idx).
            close_interval_threshold (int, optional): The maximum number of timesteps between intervals to be considered close.
                                                      Defaults to CLOSE_INTERVAL_THRESHOLD_DEFAULT.

        Returns:
            bool: True if the intervals are close to each other, False otherwise.
        """
        are_close_intervals = (
            abs(grasp_interval_1[0] - grasp_interval_2[0]) <= close_interval_threshold
        )
        are_close_intervals |= (
            abs(grasp_interval_1[1] - grasp_interval_2[0]) <= close_interval_threshold
        )
        are_close_intervals |= (
            abs(grasp_interval_1[0] - grasp_interval_2[1]) <= close_interval_threshold
        )
        are_close_intervals |= (
            abs(grasp_interval_1[1] - grasp_interval_2[1]) <= close_interval_threshold
        )
        return are_close_intervals

    @staticmethod
    def interval_close_to_demo_start(
        grasp_interval: Tuple[int, int],
        close_interval_threshold: int = CLOSE_INTERVAL_THRESHOLD_DEFAULT,
    ) -> bool:
        return grasp_interval[0] <= close_interval_threshold

    @staticmethod
    def interval_close_to_demo_end(
        grasp_interval: Tuple[int, int],
        demo_length: int,
        close_interval_threshold: int = CLOSE_INTERVAL_THRESHOLD_DEFAULT,
    ) -> bool:
        return grasp_interval[1] >= demo_length - close_interval_threshold

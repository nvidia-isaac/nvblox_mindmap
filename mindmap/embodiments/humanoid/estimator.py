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

from typing import List, Optional

import torch

from mindmap.embodiments.delay_based_estimator import DelayBasedGripperStateEstimator
from mindmap.embodiments.estimator_base import OfflineEstimatorBase, OnlineEstimatorBase
from mindmap.embodiments.humanoid.hand import (
    get_tensor_of_proximal_joints,
    is_hand_closed_instantaneous_from_proximal_joint_states,
    is_hand_open_instantaneous_from_proximal_joint_states,
)
from mindmap.embodiments.humanoid.keypose_estimation import HumanoidEmbodimentKeyposeEstimator
from mindmap.embodiments.humanoid.policy_state import HumanoidEmbodimentPolicyState
from mindmap.embodiments.humanoid.robot_state import HumanoidEmbodimentRobotState


class HumanoidEmbodimentOnlineEstimator(OnlineEstimatorBase):
    """Estimate a policy state from a robot state.

    In contrast to the offline estimator, this estimator runs in closed-loop
    so does not have access to the entire trajectory of robot states, only
    the current robot state.

    """

    def __init__(self):
        # Params
        self.steps_commanded_to_take_affect = 10
        # The delay based estimator
        # NOTE: This is optional because we need the first robot state to initialize it.
        self.left_hand_delay_based_estimator: Optional[DelayBasedGripperStateEstimator] = None
        self.right_hand_delay_based_estimator: Optional[DelayBasedGripperStateEstimator] = None

    def __call__(
        self, state: HumanoidEmbodimentRobotState, last_goal_state: HumanoidEmbodimentPolicyState
    ) -> HumanoidEmbodimentPolicyState:
        """Estimate a policy state from a robot state.

        Args:
            state (HumanoidEmbodimentRobotState): The current robot state.
            last_goal_state (HumanoidEmbodimentPolicyState): The last goal state.

        Returns:
            HumanoidEmbodimentPolicyState: The estimated policy state.
        """

        # Get the delay based estimators
        if self.left_hand_delay_based_estimator is None:
            self.left_hand_delay_based_estimator = self._get_delay_based_estimator(
                hand_joint_state=state.left_hand_joint_states
            )
        if self.right_hand_delay_based_estimator is None:
            self.right_hand_delay_based_estimator = self._get_delay_based_estimator(
                hand_joint_state=state.right_hand_joint_states
            )

        # Get the estimates from the delay based estimators
        left_hand_closedness_goal = (
            last_goal_state.left_hand_closedness if last_goal_state is not None else None
        )
        left_hand_closedness = self._update_and_get_estimate(
            hand_joint_state=state.left_hand_joint_states,
            hand_closedness_goal=left_hand_closedness_goal,
            delay_based_estimator=self.left_hand_delay_based_estimator,
        )
        right_hand_closedness_goal = (
            last_goal_state.right_hand_closedness if last_goal_state is not None else None
        )
        right_hand_closedness = self._update_and_get_estimate(
            hand_joint_state=state.right_hand_joint_states,
            hand_closedness_goal=right_hand_closedness_goal,
            delay_based_estimator=self.right_hand_delay_based_estimator,
        )

        # Construct the policy state
        return HumanoidEmbodimentPolicyState(
            W_t_W_LeftEef=state.W_t_W_LeftEef,
            q_wxyz_W_LeftEef=state.q_wxyz_W_LeftEef,
            left_hand_closedness=left_hand_closedness.float(),
            W_t_W_RightEef=state.W_t_W_RightEef,
            q_wxyz_W_RightEef=state.q_wxyz_W_RightEef,
            right_hand_closedness=right_hand_closedness.float(),
            head_yaw_rad=state.head_yaw_rad,
        )

    def _get_delay_based_estimator(
        self, hand_joint_state: torch.Tensor
    ) -> DelayBasedGripperStateEstimator:
        hand_proximal_joint_states, _ = get_tensor_of_proximal_joints(hand_joint_state.view(1, -1))
        current_closedness = is_hand_closed_instantaneous_from_proximal_joint_states(
            hand_proximal_joint_states.squeeze(0)
        )
        delay_based_estimator = DelayBasedGripperStateEstimator(
            initial_state=current_closedness,
            steps_commanded_to_take_affect=self.steps_commanded_to_take_affect,
        )
        return delay_based_estimator

    def _update_and_get_estimate(
        self,
        hand_joint_state: torch.Tensor,
        hand_closedness_goal: Optional[torch.Tensor],
        delay_based_estimator: Optional[DelayBasedGripperStateEstimator],
    ) -> torch.tensor:
        assert hand_joint_state.shape == (11,)
        if hand_closedness_goal is not None:
            assert hand_closedness_goal.shape == (1,)

        # Update
        # NOTE(alexmillane): It's important that this function is called once per step.
        last_gripper_command_float = (
            hand_closedness_goal.item() if hand_closedness_goal is not None else None
        )
        delay_based_estimator.update(last_gripper_command_float)

        # Get the current gripper state
        current_gripper_is_closed = delay_based_estimator.get_state()
        current_gripper_is_closed = torch.tensor([current_gripper_is_closed], device="cuda")
        assert current_gripper_is_closed.shape == (1,)
        return current_gripper_is_closed


class HumanoidEmbodimentOfflineEstimator(OfflineEstimatorBase):
    def __init__(self):
        super().__init__()
        # We create a keypose estimator in order to determine open-closed intervals.
        self.keypose_estimator = HumanoidEmbodimentKeyposeEstimator()

    def policy_states_from_robot_states(
        self, robot_state_vec: List[HumanoidEmbodimentRobotState], use_keyposes: bool = True
    ) -> List[HumanoidEmbodimentPolicyState]:
        """Get a vector of policy states from a vector of robot states.

        Args:
            robot_state_vec: A vector of robot states.
            use_keyposes: Whether we're in keypose mode. In keypose mode, the closedness
                of the gripper comes from the intervals between the keyposes, which can
                differ from the instantaneous state of the gripper.

        Returns:
            List[HumanoidEmbodimentPolicyState]: A vector of policy states.
        """
        # If we need them grab the openness intervals
        if use_keyposes:
            grasp_intervals = self.keypose_estimator.get_grasp_events(robot_state_vec)
            assert len(robot_state_vec) == len(grasp_intervals.left_hand_open_masks)
            assert len(robot_state_vec) == len(grasp_intervals.right_hand_open_masks)

        # Assemble the policy states
        policy_states: List[HumanoidEmbodimentPolicyState] = []
        for idx, robot_state in enumerate(robot_state_vec):
            # Determine the closedness
            if use_keyposes:
                left_hand_open = grasp_intervals.left_hand_open_masks[idx]
                right_hand_open = grasp_intervals.right_hand_open_masks[idx]
            else:
                left_hand_proximal_joint_states, _ = get_tensor_of_proximal_joints(
                    robot_state.left_hand_joint_states.unsqueeze(0)
                )
                right_hand_proximal_joint_states, _ = get_tensor_of_proximal_joints(
                    robot_state.right_hand_joint_states.unsqueeze(0)
                )
                left_hand_open = is_hand_open_instantaneous_from_proximal_joint_states(
                    left_hand_proximal_joint_states.squeeze(0)
                )
                right_hand_open = is_hand_open_instantaneous_from_proximal_joint_states(
                    right_hand_proximal_joint_states.squeeze(0)
                )
            left_hand_closedness = torch.logical_not(
                torch.tensor([left_hand_open], dtype=torch.bool)
            )
            right_hand_closedness = torch.logical_not(
                torch.tensor([right_hand_open], dtype=torch.bool)
            )
            # Construct the policy state
            policy_states.append(
                HumanoidEmbodimentPolicyState(
                    W_t_W_LeftEef=robot_state.W_t_W_LeftEef,
                    q_wxyz_W_LeftEef=robot_state.q_wxyz_W_LeftEef,
                    left_hand_closedness=left_hand_closedness,
                    W_t_W_RightEef=robot_state.W_t_W_RightEef,
                    q_wxyz_W_RightEef=robot_state.q_wxyz_W_RightEef,
                    right_hand_closedness=right_hand_closedness,
                    head_yaw_rad=robot_state.head_yaw_rad,
                )
            )
        return policy_states

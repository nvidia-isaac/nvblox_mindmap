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

from mindmap.embodiments.arm.gripper import is_gripper_closed
from mindmap.embodiments.arm.keypose_estimation import ArmEmbodimentKeyposeEstimator
from mindmap.embodiments.arm.policy_state import ArmEmbodimentPolicyState
from mindmap.embodiments.arm.robot_state import ArmEmbodimentRobotState
from mindmap.embodiments.delay_based_estimator import DelayBasedGripperStateEstimator
from mindmap.embodiments.estimator_base import OfflineEstimatorBase, OnlineEstimatorBase


class ArmEmbodimentOnlineEstimator(OnlineEstimatorBase):
    """Estimate a policy state from a robot state.

    In contrast to the offline estimator, this estimator runs in closed-loop
    so does not have access to the entire trajectory of robot states, only
    the current robot state.

    """

    def __init__(self):
        super().__init__()
        # Params
        self.steps_commanded_to_take_affect = 10
        # The delay-based estimator
        # NOTE: This is optional because we need the first robot state to initialize it,
        # so it is initialized in the __call__ method.
        self.delay_based_estimator: Optional[DelayBasedGripperStateEstimator] = None

    def __call__(
        self, state: ArmEmbodimentRobotState, last_goal_state: ArmEmbodimentPolicyState
    ) -> ArmEmbodimentPolicyState:
        """Estimate a policy state from a robot state.

        Args:
            state (ArmEmbodimentRobotState): The current robot state.
            last_goal_state (ArmEmbodimentPolicyState): The last goal state.

        Returns:
            ArmEmbodimentPolicyState: The estimated policy state.
        """
        # Initialize underlying delay-based estimator if required
        if self.delay_based_estimator is None:
            gripper_jaw_positions = state.gripper_jaw_positions
            current_closedness = is_gripper_closed(torch.tensor(gripper_jaw_positions))
            self.delay_based_estimator = DelayBasedGripperStateEstimator(
                initial_state=current_closedness,
                steps_commanded_to_take_affect=self.steps_commanded_to_take_affect,
            )
        # Update
        # NOTE(alexmillane): It's important that this function is called once per step.
        last_gripper_command = (
            last_goal_state.gripper_closedness.item() if last_goal_state is not None else None
        )
        self.delay_based_estimator.update(last_gripper_command)

        # Get the current gripper state
        current_gripper_is_closed = self.delay_based_estimator.get_state()
        current_gripper_is_closed = torch.tensor([current_gripper_is_closed], device="cuda")
        assert current_gripper_is_closed.shape == (1,)
        return ArmEmbodimentPolicyState(
            W_t_W_Eef=state.W_t_W_Eef,
            q_wxyz_W_Eef=state.q_wxyz_W_Eef,
            gripper_closedness=current_gripper_is_closed.to(torch.float),
        )


class ArmEmbodimentOfflineEstimator(OfflineEstimatorBase):
    """Estimate a vector of policy states from a vector of robot states.

    This is used in dataset generation where a vector of robot states
    originating from a dataset is used to generate a vector of policy states
    which are ground-truth for the policy during training.

    """

    def __init__(self):
        super().__init__()
        # We create a keypose estimator in order to determine open-closed intervals.
        self.keypose_estimator = ArmEmbodimentKeyposeEstimator()

    def policy_states_from_robot_states(
        self, robot_state_vec: List[ArmEmbodimentRobotState], use_keyposes: bool = True
    ) -> List[ArmEmbodimentPolicyState]:
        """Get a vector of policy states from a vector of robot states.

        Args:
            robot_state_vec: A vector of robot states.
            use_keyposes: Whether we're in keypose mode. In keypose mode, the closedness
                of the gripper comes from the intervals between the keyposes, which can
                differ from the instantaneous state of the gripper.

        Returns:
            List[ArmEmbodimentPolicyState]: A vector of policy states.
        """
        # If we need them grab the openness intervals
        if use_keyposes:
            _, gripper_open_mask = self.keypose_estimator.get_grasp_events(robot_state_vec)
            assert len(robot_state_vec) == len(gripper_open_mask)

        policy_state_vec: List[ArmEmbodimentPolicyState] = []
        for idx, robot_state in enumerate(robot_state_vec):
            # Determine the closedness
            if use_keyposes:
                gripper_closedness = torch.logical_not(
                    torch.tensor([gripper_open_mask[idx]], dtype=torch.bool)
                )
            else:
                gripper_closedness = is_gripper_closed(robot_state.gripper_jaw_positions)
            # Construct the policy state
            policy_state = ArmEmbodimentPolicyState(
                W_t_W_Eef=robot_state.W_t_W_Eef,
                q_wxyz_W_Eef=robot_state.q_wxyz_W_Eef,
                gripper_closedness=gripper_closedness,
            )
            policy_state_vec.append(policy_state)

        return policy_state_vec

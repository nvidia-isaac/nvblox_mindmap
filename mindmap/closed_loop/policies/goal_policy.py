# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import List, Optional

from isaaclab.sensors import Camera
import torch

from mindmap.cli.args import ClosedLoopAppArgs
from mindmap.closed_loop.policies.policy_base import PolicyBase
from mindmap.embodiments.arm.policy_state import ArmEmbodimentPolicyState
from mindmap.embodiments.embodiment_base import EmbodimentBase, EmbodimentType
from mindmap.embodiments.humanoid.policy_state import HumanoidEmbodimentPolicyState
from mindmap.embodiments.observation_base import ObservationBase
from mindmap.embodiments.state_base import PolicyStateBase


class GoalPolicy(PolicyBase):
    """A policy which executes a hardcoded sequence of goals.

    Args:
        args: The closed loop application arguments.
        device: The device to return action tensors on.
        goal_states: The sequence of goals to execute.
        repeat: Whether to repeat the goals after the sequence is exhausted.
    """

    def __init__(
        self,
        args: ClosedLoopAppArgs,
        device: str,
        goal_states: List[PolicyStateBase],
        repeat: bool = True,
    ):
        self.args = args
        self.device = device
        self.goal_states = goal_states
        self.current_goal_idx = 0
        self.repeat = repeat
        # Reset for good measure.
        self.reset()

    def step(self, current_state: PolicyStateBase, observation: ObservationBase) -> None:
        """Called every simulation step to update policy's internal state."""
        pass

    def get_new_goal(
        self,
        embodiment: EmbodimentBase,
        current_state: PolicyStateBase,
        observation: ObservationBase,
    ) -> List[Optional[PolicyStateBase]]:
        """Generates a goal given the current state and camera observations."""
        if self.current_goal_idx == len(self.goal_states):
            if self.repeat:
                self.current_goal_idx = 0
            else:
                return [None]
        goal = self.goal_states[self.current_goal_idx]
        self.current_goal_idx += 1
        return [goal]

    def reset(self):
        """Resets the policy's internal state."""
        self.current_goal_idx = 0


def get_dummy_policy_for_embodiment(embodiment_type: EmbodimentType, device: str) -> GoalPolicy:
    """Returns a dummy policy for a given embodiment type.

    The dummy policy is a hardcoded sequence of test goals. This policy is used for testing.

    Args:
        embodiment_type: The type of embodiment to get a dummy policy for.
        device: The device to return action tensors on.

    Returns:
        A GoalPolicy loaded with a sequence of goals for the given embodiment type.
    """
    if embodiment_type == EmbodimentType.ARM:
        # These goals move the franka back and forth between two points, which vary
        # along the y-axis in front of the robot.
        goals = [
            ArmEmbodimentPolicyState(
                W_t_W_Eef=torch.tensor([0.6, 0.25, 0.25], device=device),
                q_wxyz_W_Eef=torch.tensor([0, 1, 0, 0], device=device),
                gripper_closedness=torch.zeros((1), device=device),
            ),
            ArmEmbodimentPolicyState(
                W_t_W_Eef=torch.tensor([0.6, 0.25 - 0.2, 0.25], device=device),
                q_wxyz_W_Eef=torch.tensor([0, 1, 0, 0], device=device),
                gripper_closedness=torch.zeros((1), device=device),
            ),
        ]

    elif embodiment_type == EmbodimentType.HUMANOID:
        # These goals move both the humanoid hands up and down between two points,
        # which vary along the z-axis in front of the robot and moves the head left and right.
        goals = [
            HumanoidEmbodimentPolicyState(
                W_t_W_LeftEef=torch.tensor([-0.2236, 0.2580, 1.0964], device=device),
                q_wxyz_W_LeftEef=torch.tensor([0.5039, 0.4955, -0.5064, 0.4941], device=device),
                left_hand_closedness=torch.tensor([1.0], device=device),
                W_t_W_RightEef=torch.tensor([0.0605, 0.2517, 1.1063], device=device),
                q_wxyz_W_RightEef=torch.tensor([0.4773, 0.5318, -0.4857, 0.5034], device=device),
                right_hand_closedness=torch.tensor([0.0], device=device),
                head_yaw_rad=torch.tensor([-1.57], device=device),
            ),
            HumanoidEmbodimentPolicyState(
                W_t_W_LeftEef=torch.tensor([-0.2236, 0.2580, 1.0964 + 0.2], device=device),
                q_wxyz_W_LeftEef=torch.tensor([0.5039, 0.4955, -0.5064, 0.4941], device=device),
                left_hand_closedness=torch.tensor([0.0], device=device),
                W_t_W_RightEef=torch.tensor([0.0605 + 0.3, 0.2517, 1.1063 + 0.2], device=device),
                q_wxyz_W_RightEef=torch.tensor([0.4773, 0.5318, -0.4857, 0.5034], device=device),
                right_hand_closedness=torch.tensor([1.0], device=device),
                head_yaw_rad=torch.tensor([1.57], device=device),
            ),
        ]
    else:
        raise ValueError(f"Invalid embodiment type: {embodiment_type}")
    return GoalPolicy(args=None, device=device, goal_states=goals)

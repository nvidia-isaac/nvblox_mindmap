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

import torch

from mindmap.embodiments.humanoid.joint_indices import HumanoidJointIndices
from mindmap.embodiments.state_base import ActionBase


@dataclass
class HumanoidEmbodimentAction(ActionBase):
    W_t_W_LeftEef: torch.Tensor
    """Position of the left end-effector in the world"""

    q_wxyz_W_LeftEef: torch.Tensor
    """Orientation of the left end-effector in the world"""

    left_hand_joint_states: torch.Tensor
    """Joint states of the left hand"""

    W_t_W_RightEef: torch.Tensor
    """Position of the right end-effector in the world"""

    q_wxyz_W_RightEef: torch.Tensor
    """Orientation of the right end-effector in the world"""

    right_hand_joint_states: torch.Tensor
    """Joint states of the right hand"""

    head_yaw_rad: torch.Tensor
    """
    Yaw of the head in radians:
        - 0 rad is forward, negative is turning head clockwise
        - in range [-pi, pi)
    """

    def __post_init__(self):
        assert self.W_t_W_LeftEef.shape == (3,)
        assert self.q_wxyz_W_LeftEef.shape == (4,)
        assert self.left_hand_joint_states.shape == (11,)
        assert self.W_t_W_RightEef.shape == (3,)
        assert self.q_wxyz_W_RightEef.shape == (4,)
        assert self.right_hand_joint_states.shape == (11,)
        assert self.head_yaw_rad.shape == (1,)
        assert self.head_yaw_rad >= -torch.pi and self.head_yaw_rad < torch.pi

    def to_tensor(self, include_head_yaw) -> torch.Tensor:
        # Combine the eef poses
        eef_poses_tensor = torch.cat(
            (self.W_t_W_LeftEef, self.q_wxyz_W_LeftEef, self.W_t_W_RightEef, self.q_wxyz_W_RightEef)
        )
        # Combine the hands first
        combined_hands_tensor = torch.zeros((22,), device=self.left_hand_joint_states.device)
        combined_hands_tensor[
            HumanoidJointIndices.left_joints_in_combined_hands_tensor_indices
        ] = self.left_hand_joint_states
        combined_hands_tensor[
            HumanoidJointIndices.right_joints_in_combined_hands_tensor_indices
        ] = self.right_hand_joint_states
        # Build the full action tensor
        actions_tensor = eef_poses_tensor
        if include_head_yaw:
            actions_tensor = torch.cat((actions_tensor, self.head_yaw_rad))
        actions_tensor = torch.cat((actions_tensor, combined_hands_tensor))
        assert actions_tensor.shape == (
            self.state_size() if include_head_yaw else self.state_size() - 1,
        )
        return actions_tensor

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> HumanoidEmbodimentAction:
        assert tensor.dim() == 1
        assert tensor.shape[0] == HumanoidEmbodimentAction.state_size()
        # NOTE(alexmillane): This order is almost certainly wrong. It is definitely incorrect
        # with respect to the order that they appear in the environment.
        # See: mindmap/embodiments/humanoid/joint_indices.py
        # for the order used to extract the hand joint states from the env.
        # TODO(alexmillane): Fix this.
        return HumanoidEmbodimentAction(
            W_t_W_LeftEef=tensor[0:3],
            q_wxyz_W_LeftEef=tensor[3:7],
            W_t_W_RightEef=tensor[7:10],
            q_wxyz_W_RightEef=tensor[10:14],
            head_yaw_rad=tensor[14:15],
            left_hand_joint_states=tensor[15:26],
            right_hand_joint_states=tensor[26:37],
        )

    @staticmethod
    def state_size() -> int:
        return 37

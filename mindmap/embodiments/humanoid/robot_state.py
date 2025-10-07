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

from mindmap.embodiments.state_base import RobotStateBase

NUM_HAND_JOINTS = 11


@dataclass
class HumanoidEmbodimentRobotState(RobotStateBase):
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
        assert self.left_hand_joint_states.shape == (NUM_HAND_JOINTS,)
        assert self.W_t_W_RightEef.shape == (3,)
        assert self.q_wxyz_W_RightEef.shape == (4,)
        assert self.right_hand_joint_states.shape == (NUM_HAND_JOINTS,)
        assert self.head_yaw_rad.shape == (1,)
        assert self.head_yaw_rad >= -torch.pi and self.head_yaw_rad < torch.pi

    def to_tensor(self) -> torch.Tensor:
        return torch.cat(
            (
                self.W_t_W_LeftEef,
                self.q_wxyz_W_LeftEef,
                self.left_hand_joint_states,
                self.W_t_W_RightEef,
                self.q_wxyz_W_RightEef,
                self.right_hand_joint_states,
                self.head_yaw_rad,
            )
        )

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> HumanoidEmbodimentRobotState:
        assert tensor.dim() == 1
        if tensor.shape[0] == HumanoidEmbodimentRobotState.state_size() - 1:
            # TODO(remos): Remove this once we have updated all the data.
            print(
                "WARNING: Tensor is missing a value. "
                "Assuming this is old data without head yaw. Adding zero head yaw."
            )
            tensor = torch.cat((tensor, torch.zeros(1)))
        assert tensor.shape[0] == HumanoidEmbodimentRobotState.state_size()
        # NOTE(alexmillane, 2025.05.16): This is a different order to the action tensor.
        # Maybe it should be unified. Leaving as is for now.
        # NOTE(alexmillane, 2025.05.27): This is a different order to IsaacLab.
        # Potentially this will cause issues in the future.
        return HumanoidEmbodimentRobotState(
            W_t_W_LeftEef=tensor[0:3],
            q_wxyz_W_LeftEef=tensor[3:7],
            left_hand_joint_states=tensor[7:18],
            W_t_W_RightEef=tensor[18:21],
            q_wxyz_W_RightEef=tensor[21:25],
            right_hand_joint_states=tensor[25:36],
            head_yaw_rad=tensor[36:37],
        )

    @staticmethod
    def state_size() -> int:
        return 37

    @staticmethod
    def num_joints_per_hand() -> int:
        return NUM_HAND_JOINTS

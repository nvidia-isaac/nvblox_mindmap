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

from mindmap.embodiments.state_base import PolicyStateBase


@dataclass
class HumanoidEmbodimentPolicyState(PolicyStateBase):
    W_t_W_LeftEef: torch.Tensor
    """Position of the left end-effector in the world"""

    q_wxyz_W_LeftEef: torch.Tensor
    """Orientation of the left end-effector in the world"""

    left_hand_closedness: torch.Tensor
    """Closedness of the left hand"""

    W_t_W_RightEef: torch.Tensor
    """Position of the right end-effector in the world"""

    q_wxyz_W_RightEef: torch.Tensor
    """Orientation of the right end-effector in the world"""

    right_hand_closedness: torch.Tensor
    """Closedness of the right hand"""

    head_yaw_rad: torch.Tensor
    """
    Yaw of the head in radians:
        - 0 rad is forward, negative is turning head clockwise
        - in range [-pi, pi)
    """

    def __post_init__(self):
        assert self.W_t_W_LeftEef.shape == (3,)
        assert self.q_wxyz_W_LeftEef.shape == (4,)
        assert self.left_hand_closedness.shape == (1,)
        assert self.W_t_W_RightEef.shape == (3,)
        assert self.q_wxyz_W_RightEef.shape == (4,)
        assert self.right_hand_closedness.shape == (1,)
        assert self.head_yaw_rad.shape == (1,)
        assert self.head_yaw_rad >= -torch.pi and self.head_yaw_rad < torch.pi

    def to_tensor(self) -> torch.Tensor:
        return torch.cat(
            (
                self.W_t_W_LeftEef,
                self.q_wxyz_W_LeftEef,
                self.left_hand_closedness,
                self.W_t_W_RightEef,
                self.q_wxyz_W_RightEef,
                self.right_hand_closedness,
                self.head_yaw_rad,
            )
        )

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> HumanoidEmbodimentPolicyState:
        assert tensor.dim() == 1
        assert tensor.shape[0] == HumanoidEmbodimentPolicyState.state_size()
        return HumanoidEmbodimentPolicyState(
            W_t_W_LeftEef=tensor[0:3],
            q_wxyz_W_LeftEef=tensor[3:7],
            left_hand_closedness=tensor[7:8],
            W_t_W_RightEef=tensor[8:11],
            q_wxyz_W_RightEef=tensor[11:15],
            right_hand_closedness=tensor[15:16],
            head_yaw_rad=tensor[16:17],
        )

    @staticmethod
    def state_size() -> int:
        return 17

    @staticmethod
    def split_gripper_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Args:
        - tensor: [B, nhist, 8 * n_grippers + 1]
            For this embodiment n_grippers is 2
            We assume that the first 8 elements are the left hand
            and the last 8 elements are the right hand.

        Returns:
        - torch.Tensor: [B, nhist, n_grippers, 8]
            For this embodiment n_grippers is 2"""
        assert tensor.dim() == 3
        assert tensor.shape[2] == HumanoidEmbodimentPolicyState.state_size()
        # The first 8 elements are the left hand
        left_hand_tensor = tensor[..., :8]  # Shape: [B,Nhist,8]
        right_hand_tensor = tensor[..., 8:16]  # Shape: [B,Nhist,8]
        # Stack the tensors along a new dimension to get shape [B,Nhist,2,8]
        return torch.stack((left_hand_tensor, right_hand_tensor), dim=-2)

    @staticmethod
    def split_head_yaw_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Args:
        - tensor: [B, nhist, 8 * n_grippers + 1]

        Returns:
        - torch.Tensor: [B, nhist, 1]"""
        assert tensor.dim() == 3
        assert tensor.shape[2] == HumanoidEmbodimentPolicyState.state_size()
        head_yaw_tensor = tensor[..., 16:17]  # Shape: [B,Nhist,1]
        assert torch.all(head_yaw_tensor >= -torch.pi) and torch.all(head_yaw_tensor < torch.pi)

        return head_yaw_tensor

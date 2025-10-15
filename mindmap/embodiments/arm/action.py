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

from mindmap.embodiments.state_base import ActionBase


@dataclass
class ArmEmbodimentAction(ActionBase):
    W_t_W_Eef: torch.Tensor
    """Position of the end-effector in the world"""

    q_wxyz_W_Eef: torch.Tensor
    """Orientation of the end-effector in the world"""

    gripper_command: torch.Tensor
    """The command to the gripper. 1.0 is open, -1.0 is closed."""

    def __post_init__(self):
        assert self.W_t_W_Eef.shape == (3,)
        assert self.q_wxyz_W_Eef.shape == (4,)
        assert self.gripper_command.shape == (1,)

    def to_tensor(self) -> torch.Tensor:
        return torch.cat((self.W_t_W_Eef, self.q_wxyz_W_Eef, self.gripper_command))

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> ArmEmbodimentAction:
        assert tensor.dim() == 1
        assert tensor.shape[0] == ArmEmbodimentAction.state_size()
        return ArmEmbodimentAction(
            W_t_W_Eef=tensor[0:3], q_wxyz_W_Eef=tensor[3:7], gripper_command=tensor[7:8]
        )

    @staticmethod
    def state_size() -> int:
        return 8

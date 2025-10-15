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

import torch

from mindmap.embodiments.arm.action import ArmEmbodimentAction
from mindmap.embodiments.arm.policy_state import ArmEmbodimentPolicyState
from mindmap.embodiments.controller_base import ControllerBase

ARM_CLOSEDNESS_THRESHOLD = 0.5


class ArmEmbodimentController(ControllerBase):
    def __call__(self, state: ArmEmbodimentPolicyState) -> ArmEmbodimentAction:
        gripper_closedness = state.gripper_closedness.item()
        assert gripper_closedness >= 0.0
        assert gripper_closedness <= 1.0
        if gripper_closedness > ARM_CLOSEDNESS_THRESHOLD:
            # Close gripper.
            gripper_command = -1.0
        else:
            # Open gripper.
            gripper_command = 1.0
        return ArmEmbodimentAction(
            W_t_W_Eef=state.W_t_W_Eef,
            q_wxyz_W_Eef=state.q_wxyz_W_Eef,
            gripper_command=torch.tensor([gripper_command], device=state.W_t_W_Eef.device),
        )

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

from mindmap.embodiments.controller_base import ControllerBase
from mindmap.embodiments.humanoid.action import HumanoidEmbodimentAction
from mindmap.embodiments.humanoid.joint_indices import HumanoidJointIndices
from mindmap.embodiments.humanoid.policy_state import HumanoidEmbodimentPolicyState

HUMANOID_CLOSEDNESS_THRESHOLD = 0.5

CLOSED_HAND_JOINT_STATES = {
    "index_proximal_joint": -1.35,
    "middle_proximal_joint": -1.57,
    "pinky_proximal_joint": -1.57,
    "ring_proximal_joint": -1.57,
    "thumb_proximal_yaw_joint": -1.57,
    "index_intermediate_joint": 0.35,
    "middle_intermediate_joint": 0.18,
    "pinky_intermediate_joint": -0.60,
    "ring_intermediate_joint": -0.72,
    "thumb_proximal_pitch_joint": 1.11,
    "thumb_distal_joint": -0.24,
}

OPEN_HAND_JOINT_STATES = {
    "index_proximal_joint": -0.00,
    "middle_proximal_joint": 0.00,
    "pinky_proximal_joint": 0.00,
    "ring_proximal_joint": 0.00,
    "thumb_proximal_yaw_joint": -1.57,
    "index_intermediate_joint": 0.00,
    "middle_intermediate_joint": -0.00,
    "pinky_intermediate_joint": 0.00,
    "ring_intermediate_joint": 0.00,
    "thumb_proximal_pitch_joint": 0.0,
    "thumb_distal_joint": 0.43,
}


class HumanoidEmbodimentController(ControllerBase):
    def __init__(self):
        super().__init__()
        # Check that the joint names/orders haven't changed
        assert list(CLOSED_HAND_JOINT_STATES.keys()) == list(
            HumanoidJointIndices.within_hand_joint_name_to_idx_map.keys()
        )
        assert list(OPEN_HAND_JOINT_STATES.keys()) == list(
            HumanoidJointIndices.within_hand_joint_name_to_idx_map.keys()
        )
        # Convert the closed/open joint state to a tensor
        self.closed_hand_joint_states = torch.tensor(list(CLOSED_HAND_JOINT_STATES.values()))
        self.open_hand_joint_states = torch.tensor(list(OPEN_HAND_JOINT_STATES.values()))

    def get_hand_joint_states(self, closedness: float, device: torch.device) -> torch.Tensor:
        # Get the hand joint states
        if closedness > HUMANOID_CLOSEDNESS_THRESHOLD:
            return self.closed_hand_joint_states.to(device=device)
        else:
            return self.open_hand_joint_states.to(device=device)

    def __call__(self, state: HumanoidEmbodimentPolicyState) -> HumanoidEmbodimentAction:
        # Get the hand joint states from closedness
        left_hand_joint_states = self.get_hand_joint_states(
            state.left_hand_closedness, device=state.W_t_W_LeftEef.device
        )
        right_hand_joint_states = self.get_hand_joint_states(
            state.right_hand_closedness, device=state.W_t_W_RightEef.device
        )
        # Combine to a full action
        return HumanoidEmbodimentAction(
            W_t_W_LeftEef=state.W_t_W_LeftEef,
            q_wxyz_W_LeftEef=state.q_wxyz_W_LeftEef,
            left_hand_joint_states=left_hand_joint_states,
            W_t_W_RightEef=state.W_t_W_RightEef,
            q_wxyz_W_RightEef=state.q_wxyz_W_RightEef,
            right_hand_joint_states=right_hand_joint_states,
            head_yaw_rad=state.head_yaw_rad,
        )

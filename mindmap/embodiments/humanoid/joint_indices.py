# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from enum import Enum
from typing import Dict, Optional


class HandSide(Enum):
    LEFT = "left"
    RIGHT = "right"


class _HumanoidJointIndices:
    # A dictionary mapping joint names to their indices.
    # NOTE(alexmillane): This is take from the task definition: nutpour_gr1t2_closedloop_env_cfg.py
    # It would be better to import it, however that requires have IsaacLab started, and
    # we need these constants in contexts where IsaacLab is not started, so here we duplicate
    # it.
    joint_names_dict: Dict[str, int] = {
        # arm joints
        "left_shoulder_pitch_joint": 0,
        "right_shoulder_pitch_joint": 1,
        "left_shoulder_roll_joint": 2,
        "right_shoulder_roll_joint": 3,
        "left_shoulder_yaw_joint": 4,
        "right_shoulder_yaw_joint": 5,
        "left_elbow_pitch_joint": 6,
        "right_elbow_pitch_joint": 7,
        "left_wrist_yaw_joint": 8,
        "right_wrist_yaw_joint": 9,
        "left_wrist_roll_joint": 10,
        "right_wrist_roll_joint": 11,
        "left_wrist_pitch_joint": 12,
        "right_wrist_pitch_joint": 13,
        # hand joints
        "L_index_proximal_joint": 14,
        "L_middle_proximal_joint": 15,
        "L_pinky_proximal_joint": 16,
        "L_ring_proximal_joint": 17,
        "L_thumb_proximal_yaw_joint": 18,
        "R_index_proximal_joint": 19,
        "R_middle_proximal_joint": 20,
        "R_pinky_proximal_joint": 21,
        "R_ring_proximal_joint": 22,
        "R_thumb_proximal_yaw_joint": 23,
        "L_index_intermediate_joint": 24,
        "L_middle_intermediate_joint": 25,
        "L_pinky_intermediate_joint": 26,
        "L_ring_intermediate_joint": 27,
        "L_thumb_proximal_pitch_joint": 28,
        "R_index_intermediate_joint": 29,
        "R_middle_intermediate_joint": 30,
        "R_pinky_intermediate_joint": 31,
        "R_ring_intermediate_joint": 32,
        "R_thumb_proximal_pitch_joint": 33,
        "L_thumb_distal_joint": 34,
        "R_thumb_distal_joint": 35,
    }

    def __init__(self):
        # Constants
        self.num_joints_per_hand = len(
            [joint_name for joint_name in self.joint_names_dict.keys() if "L" in joint_name]
        )
        # The indices of the hand joints *within* the last 22 joint indices.
        self.hand_joint_name_to_idx_map = self._get_hand_joint_name_to_idx_map()
        self.left_hand_name_to_idx_map = self._get_hand_joint_name_to_idx_map(HandSide.LEFT)
        self.right_hand_name_to_idx_map = self._get_hand_joint_name_to_idx_map(HandSide.RIGHT)
        self.within_hand_joint_name_to_idx_map = self._get_within_hand_joint_name_to_idx_map()
        # These are lists of indices into the combined hands tensor for each of the hand joints.
        self.left_joints_in_combined_hands_tensor_indices = list(
            self.left_hand_name_to_idx_map.values()
        )
        self.right_joints_in_combined_hands_tensor_indices = list(
            self.right_hand_name_to_idx_map.values()
        )

    def _get_hand_joint_name_to_idx_map(
        self, hand_side: Optional[HandSide] = None
    ) -> Dict[str, int]:
        finger_joint_names_dict = {
            key: idx for key, idx in self.joint_names_dict.items() if "L" in key or "R" in key
        }
        min_finger_joint_indices = min(finger_joint_names_dict.values())
        finger_joint_names_dict = {
            key: (idx - min_finger_joint_indices) for key, idx in finger_joint_names_dict.items()
        }
        if hand_side is HandSide.LEFT:
            finger_joint_names_dict = {
                key: idx for key, idx in finger_joint_names_dict.items() if "L" in key
            }
        elif hand_side is HandSide.RIGHT:
            finger_joint_names_dict = {
                key: idx for key, idx in finger_joint_names_dict.items() if "R" in key
            }
        return finger_joint_names_dict

    def _get_within_hand_joint_name_to_idx_map(self) -> Dict[str, int]:
        within_hand_joint_to_idx_map = {}
        left_hand_name_to_idx_map = self._get_hand_joint_name_to_idx_map(HandSide.LEFT)
        for idx, joint_name in enumerate(left_hand_name_to_idx_map.keys()):
            within_hand_joint_to_idx_map[joint_name.strip("L_")] = idx
        return within_hand_joint_to_idx_map


HumanoidJointIndices = _HumanoidJointIndices()

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import torch

from mindmap.embodiments.humanoid.joint_indices import HumanoidJointIndices
from mindmap.embodiments.humanoid.robot_state import HumanoidEmbodimentRobotState

# The threshold below which we consider the hand to be closed.
# NOTE(remos): We only allow fully open or closed hands.
#              Fully open is 0.0, while fully closed is -1.57.
#              Selecting a threshold near the open value ensures
#              that we detect closedness even when the hand is
#              holding an object and can't fully close.
CLOSED_THRESHOLD = -0.4


def is_hand_closed_instantaneous_from_proximal_joint_states(
    proximal_joint_states: torch.Tensor,
) -> bool:
    """Check if the hand is closed instantaneously from the proximal joint states.

    Note that the "instantaneous" here is to differentiate this function from other methods that
    use the state history, either for hysteresis or delay-based estimation of the closedness.

    Args:
        proximal_joint_states: A tensor (num_proximal_joints,) containing the proximal joint states.

    Returns:
        True if the hand is closed.

    """
    assert proximal_joint_states.ndim == 1
    assert proximal_joint_states.shape[0] < HumanoidEmbodimentRobotState.num_joints_per_hand()
    return bool(torch.any(proximal_joint_states < CLOSED_THRESHOLD).item())


def is_hand_open_instantaneous_from_proximal_joint_states(
    proximal_joint_states: torch.Tensor,
) -> bool:
    """Check if the hand is open instantaneously from the proximal joint states.

    Note that the "instantaneous" here is to differentiate this function from other methods that
    use the state history, either for hysteresis or delay-based estimation of the closedness.

    Args:
        proximal_joint_states: A tensor (num_proximal_joints,) containing the proximal joint states.

    Returns:
        True if the hand is open.

    """
    return not is_hand_closed_instantaneous_from_proximal_joint_states(proximal_joint_states)


def get_tensor_of_proximal_joints(
    one_hand_joint_states: torch.Tensor, excluded_joint_strings: List[str] = ["thumb", "index"]
) -> Tuple[torch.Tensor, List[str]]:
    """Returns a tensor containing the joint states of the proximal joints of a hand.

    Args:
        one_hand_joint_states: A tensor of shape (num_samples, num_joints) containing the joint states of a hand.
        excluded_joint_strings: A list of strings to exclude from the proximal joints.

    Returns:
        A tensor of shape (num_samples, num_proximal_joints) containing the joint states of the proximal joints of a hand.
        A list of the proximal joint names.
    """
    assert one_hand_joint_states.ndim == 2
    assert one_hand_joint_states.shape[1] == HumanoidJointIndices.num_joints_per_hand
    proximal_joint_to_idx_map = {}
    for joint_name, idx in HumanoidJointIndices.within_hand_joint_name_to_idx_map.items():
        if "proximal" in joint_name:
            if not any(s in joint_name for s in excluded_joint_strings):
                proximal_joint_to_idx_map[joint_name] = idx
    proximal_joint_indices = list(proximal_joint_to_idx_map.values())
    return one_hand_joint_states[:, proximal_joint_indices], list(proximal_joint_to_idx_map.keys())

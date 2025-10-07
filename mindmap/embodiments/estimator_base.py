# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from abc import ABC, abstractmethod
from typing import List

from mindmap.embodiments.state_base import PolicyStateBase, RobotStateBase


class OnlineEstimatorBase(ABC):
    """Base class for online estimators.

    Online estimators convert robot states to policy states during policy execution.

    """

    @abstractmethod
    def __call__(self, state: RobotStateBase, last_goal_state: PolicyStateBase) -> PolicyStateBase:
        """Convert a robot state to a policy state."""
        raise NotImplementedError


class OfflineEstimatorBase(ABC):
    """Base class for offline estimators.

    Offline estimators convert robot states to policy states during training.

    """

    @abstractmethod
    def policy_states_from_robot_states(
        self, robot_state_vec: List[RobotStateBase], use_keyposes: bool = True
    ) -> List[PolicyStateBase]:
        """Convert a vector of robot states to a vector of policy states."""
        raise NotImplementedError

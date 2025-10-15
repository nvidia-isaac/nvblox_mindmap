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
from enum import Enum

import gymnasium as gym

from mindmap.embodiments.state_base import ActionBase, PolicyStateBase


class ControllerBase(ABC):
    """Base class for controllers.

    Controllers convert policy states to actions.

    """

    @abstractmethod
    def __call__(self, state: PolicyStateBase) -> ActionBase:
        """Convert a policy state to an action."""
        raise NotImplementedError

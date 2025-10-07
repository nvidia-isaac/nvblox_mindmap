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

from mindmap.embodiments.embodiment_base import EmbodimentBase
from mindmap.embodiments.observation_base import ObservationBase
from mindmap.embodiments.state_base import PolicyStateBase


class PolicyBase(ABC):
    """A base class for all policies."""

    @abstractmethod
    def step(self, current_state: PolicyStateBase, observation: ObservationBase) -> None:
        """Called every simulation step to update policy's internal state."""
        pass

    @abstractmethod
    def get_new_goal(
        self,
        embodiment: EmbodimentBase,
        current_state: PolicyStateBase,
        observation: ObservationBase,
    ) -> List[PolicyStateBase]:
        """Generates a goal given the current state and observations."""
        pass

    @abstractmethod
    def reset(self):
        """Resets the policy's internal state."""
        pass

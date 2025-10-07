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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Deque, List, Type, TypeVar

import torch

T = TypeVar("T", bound="StateBase")


def state_tensor_from_history(history: List[StateBase] | Deque[StateBase]) -> torch.Tensor:
    """Converts a list of states to a single tensor.

    Args:
        history (List[State] | Deque[State]): N states

    Returns:
        torch.Tensor: NxM tensor.
        N - history length
        M - state size
    """
    history_length = len(history)
    # Set dtype to float32
    state_history_tuple = tuple([state.to_tensor().to(torch.float32) for state in history])
    # TODO(Vik): Check if view need the view to be 1, history_length, -1
    states = torch.stack(state_history_tuple).view(1, history_length, -1)
    return states


def state_tensor_from_history_list(history: List[List[StateBase]]) -> List[torch.Tensor]:
    """Converts a list of lists of states to a list of tensors.

    Args:
        history (List[List[StateBase]]): N lists of states

    Returns:
        List[torch.Tensor]: N list of M tensors.
    """
    # TODO(Vik): Check if this squeeze is needed
    return [state_tensor_from_history(history[i]).squeeze(0) for i in range(len(history))]


@dataclass
class StateBase(ABC):
    """State base class.

    Enforces convertability to and from a tensor.

    """

    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def from_tensor(tensor: torch.Tensor) -> StateBase:
        pass

    @abstractmethod
    def state_size(self) -> int:
        pass

    @classmethod
    def history_from_tensor(cls: Type[T], tensor: torch.Tensor) -> List[T]:
        """Converts a tensor of states to a list of states.

        Args:
            tensor (torch.Tensor): A tensor of shape (1, N, M) where:
                N - number of states in history
                M - state size

        Returns:
            List[T]: A list of N states of the child class type
        """
        assert tensor.dim() == 3
        assert tensor.shape[0] == 1
        # Print the values of the tensor if false
        assert (
            tensor.shape[2] == cls.state_size()
        ), f"State size mismatch: {tensor.shape[2]} != {cls.state_size()}"
        num_states = tensor.shape[1]
        states = []
        for state_idx in range(num_states):
            states.append(cls.from_tensor(tensor[0, state_idx, :]))
        return states


class RobotStateBase(StateBase):
    """Embodiment state base class."""

    pass


class PolicyStateBase(StateBase):
    """Policy state base class."""

    pass


class ActionBase(StateBase):
    """Action base class."""

    pass

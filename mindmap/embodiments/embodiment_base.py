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
from typing import List, Optional, Tuple

import gymnasium as gym
import torch

from mindmap.embodiments.controller_base import ControllerBase
from mindmap.embodiments.estimator_base import OfflineEstimatorBase, OnlineEstimatorBase
from mindmap.embodiments.keypose_estimation_base import KeyposeOnlineEstimatorBase
from mindmap.embodiments.observation_base import ObservationBase
from mindmap.embodiments.state_base import ActionBase, PolicyStateBase, RobotStateBase


class EmbodimentType(Enum):
    """The type of embodiment."""

    ARM = "arm"
    HUMANOID = "humanoid"


class EmbodimentBase(ABC):
    """Base class for Perceptive IL embodiments. The child classes define the
    end effectors and cameras of different robot embodiments.

    Args:
        env (gym.wrappers.common.OrderEnforcing): The environment instance.
    """

    # Class variable to enforce state type definition
    robot_state_type: type[RobotStateBase] = None
    policy_state_type: type[PolicyStateBase] = None
    action_type: type[ActionBase] = None
    controller_type: type[ControllerBase] = None
    online_estimator_type: type[OnlineEstimatorBase] = None
    offline_estimator_type: type[OfflineEstimatorBase] = None
    observation_type: type[ObservationBase] = None
    keypose_estimator_type: type[KeyposeOnlineEstimatorBase] = None

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._verify_embodiment_types()
        # Initialize the controller
        self.controller = self.controller_type()
        assert self.controller is not None, "Controller must be defined in embodiment subclass"
        # Initialize the estimators
        self.online_estimator = self.online_estimator_type()
        assert (
            self.online_estimator is not None
        ), "Online estimator must be defined in embodiment subclass"
        self.offline_estimator = self.offline_estimator_type()
        assert (
            self.offline_estimator is not None
        ), "Offline estimator must be defined in embodiment subclass"
        # Initialize the keypose estimator
        self.keypose_estimator = self.keypose_estimator_type()
        assert (
            self.keypose_estimator is not None
        ), "Keypose estimator must be defined in embodiment subclass"

    def _verify_embodiment_types(self):
        # Verify that the subclass has defined its state type
        required_types = [
            self.robot_state_type,
            self.policy_state_type,
            self.action_type,
            self.controller_type,
            self.online_estimator_type,
            self.offline_estimator_type,
            self.observation_type,
            self.keypose_estimator_type,
        ]
        parent_class_types = [
            RobotStateBase,
            PolicyStateBase,
            ActionBase,
            ControllerBase,
            OnlineEstimatorBase,
            OfflineEstimatorBase,
            ObservationBase,
            KeyposeOnlineEstimatorBase,
        ]
        member_names = [
            "robot_state_type",
            "policy_state_type",
            "action_type",
            "controller_type",
            "online_estimator_type",
            "offline_estimator_type",
            "observation_type",
            "keypose_estimator_type",
        ]
        for required_type, parent_class_type, member_name in zip(
            required_types, parent_class_types, member_names
        ):
            if required_type is None:
                raise NotImplementedError(
                    f"Class {self.__class__.__name__} must define {member_name}"
                )
            if not issubclass(required_type, parent_class_type):
                raise TypeError(
                    f"Class {self.__class__.__name__}'s {member_name} must be a subclass of {parent_class_type.__name__}"
                )

    @abstractmethod
    def get_robot_state(self, env: gym.wrappers.common.OrderEnforcing) -> RobotStateBase:
        raise NotImplementedError

    @abstractmethod
    def get_observation(self, env: gym.wrappers.common.OrderEnforcing) -> ObservationBase:
        raise NotImplementedError

    @abstractmethod
    def is_goal_reached(
        self,
        current_state: PolicyStateBase,
        goal_state: PolicyStateBase,
        print_errors: bool = False,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def add_intermediate_goals(
        self,
        current_state: PolicyStateBase,
        goal_state: PolicyStateBase,
    ) -> Tuple[List[PolicyStateBase], List[bool]]:
        raise NotImplementedError

    @abstractmethod
    def visualize_robot_state(
        self, robot_state: RobotStateBase, goal_state: Optional[PolicyStateBase] = None
    ):
        raise NotImplementedError

    @abstractmethod
    def get_policy_state_tensor_from_model_prediction(
        self, trajectory_pred: torch.Tensor, head_yaw_pred: torch.Tensor
    ) -> torch.Tensor:
        """Converts a model prediction to a policy state tensor."""
        raise NotImplementedError

    def get_action_from_policy_state(self, policy_state: PolicyStateBase) -> ActionBase:
        """Converts a policy state to an action."""
        return self.controller(policy_state)

    @abstractmethod
    def convert_action_to_tensor(self, action: ActionBase) -> torch.Tensor:
        """Converts an action to a tensor."""
        raise NotImplementedError

    def get_policy_state_from_embodiment_state(
        self, state: RobotStateBase, last_goal_state: PolicyStateBase
    ) -> PolicyStateBase:
        """Converts an embodiment state to a policy state."""
        return self.online_estimator(state, last_goal_state)

    def get_number_of_items_in_gripper_prediction(self):
        """Returns the number of items in the gripper prediction. This is always number of [grippers, number of prediction outputs]"""

        return [1, 0]

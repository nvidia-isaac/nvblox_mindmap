# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import torch

from mindmap.closed_loop.evaluators.evaluator_base import EvaluatorBase
from mindmap.embodiments.arm.gripper import is_gripper_closed
from mindmap.embodiments.arm.robot_state import ArmEmbodimentRobotState


class ArmEvaluatorBase(EvaluatorBase):
    """
    Base class for all evaluators that are used for the arm.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _gripper_is_open(self, observed_state: ArmEmbodimentRobotState) -> bool:
        """
        Determines if the gripper is open based on the observed state and a closeness threshold.

        Args:
            observed_state (State): The observed state.

        Returns:
            bool: True if the gripper is open, False otherwise.
        """
        gripper_pos = observed_state.gripper_jaw_positions
        return torch.logical_not(is_gripper_closed(torch.tensor(gripper_pos)))

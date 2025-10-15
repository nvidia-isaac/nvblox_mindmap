# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import Optional


class DelayBasedGripperStateEstimator:
    """A class that estimates the binary state of a gripper (open/closed).

    Once a gripper is commanded to change state for a certain amount of time, the gripper state
    will take on the new value.

    Args:
        steps_commanded_to_take_affect (int, optional): Number of consecutive steps a new state must be
            commanded before taking effect. Defaults to 10.

    """

    def __init__(self, initial_state: bool, steps_commanded_to_take_affect: int = 10):
        # Params
        self.steps_commanded_to_take_affect = steps_commanded_to_take_affect
        # State
        self.current_binarized_state: bool = initial_state
        self.last_command: Optional[bool] = None
        self.steps_commanded: int = 0

    def update(self, command_float: Optional[float] = None) -> None:
        """Updates the gripper state estimator.

        Args:
            command_float (float): The commanded closedness of the gripper.
        """
        if command_float is None:
            return
        # Float -> bool
        command = self._binarize(command_float)
        # If the first call to this function, just update the last commanded state.
        if self.last_command == None:
            self.last_command = command
        # Else look how long the new state has been commanded.
        else:
            # Check if the commanded state has changed
            if command == self.last_command:
                self.steps_commanded += 1
            else:
                self.steps_commanded = 0
            self.last_command = command
            # If long enough, take the new state
            if self.steps_commanded > self.steps_commanded_to_take_affect:
                self.current_binarized_state = command

    def get_state(self) -> bool:
        """Returns the currently estimated binarized state."""
        return self.current_binarized_state

    def _binarize(self, command_float: float) -> bool:
        """Binarizes a float command."""
        return command_float > 0.5

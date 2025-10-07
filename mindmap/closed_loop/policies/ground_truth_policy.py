# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import List, Optional

from isaaclab.sensors import Camera
import torch

from mindmap.cli.args import ClosedLoopAppArgs
from mindmap.closed_loop.gt_goals import get_gt_goals
from mindmap.closed_loop.policies.policy_base import PolicyBase
from mindmap.data_loading.data_types import includes_rgb
from mindmap.embodiments.embodiment_base import EmbodimentBase
from mindmap.embodiments.observation_base import ObservationBase
from mindmap.embodiments.state_base import PolicyStateBase
from mindmap.image_processing.image_conversions import convert_rgb_to_model_input
from mindmap.visualization.visualizer import Visualizer


class GroundTruthPolicy(PolicyBase):
    """A policy which executes the ground truth goals."""

    def __init__(self, args: ClosedLoopAppArgs, device: str):
        self.args = args
        self.device = device
        # State
        self.gt_goals_list = None
        self.goal_idx = 0
        # Visualizer
        self.visualizer = None
        if self.args.visualize:
            self.visualizer = Visualizer(args)

        # Reset for good measure.
        self.reset()

    def init_for_demo(self, demo_name: str, embodiment: EmbodimentBase) -> None:
        self.gt_goals_list = get_gt_goals(self.args, demo_name, embodiment, self.device)

    def step(self, current_state: PolicyStateBase, observation: ObservationBase) -> None:
        """Called every simulation step to update policy's internal state."""
        pass

    def get_new_goal(
        self,
        embodiment: EmbodimentBase,
        current_state: PolicyStateBase,
        observation: ObservationBase,
    ) -> List[Optional[PolicyStateBase]]:
        """Generates a goal given the current state and camera observations."""
        assert (
            self.gt_goals_list is not None
        ), "GT goals not initialized. Need to call init_for_demo() first."
        # If we're out of goals, indicate to the runner that we're done by returning None.
        if self.goal_idx >= len(self.gt_goals_list):
            return [None]
        goal_state = self.gt_goals_list[self.goal_idx]
        self.goal_idx += 1
        if self.args.visualize:
            self._visualize(observation)
        return [goal_state]

    def reset(self):
        """Resets the policy's internal state."""
        self.gt_goals_list = None
        self.goal_idx = 0

    def _visualize(self, observation: ObservationBase):
        """Visualize the current observation."""
        assert self.visualizer is not None
        cams = observation.get_cameras()

        if includes_rgb(self.args.data_type):
            rgb_stack = [
                convert_rgb_to_model_input(cam.get_rgb().squeeze(0)).unsqueeze(0)
                for cam in cams.values()
            ]
            sample = {"rgbs": torch.stack(rgb_stack, dim=1)}
        self.visualizer.visualize(sample, self.args.data_type)
        if not self.args.disable_visualizer_wait_on_key:
            self.visualizer.run_until_space_pressed()

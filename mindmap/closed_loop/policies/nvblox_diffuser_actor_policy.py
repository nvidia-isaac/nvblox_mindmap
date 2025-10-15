# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from collections import deque
import copy
import os
from typing import Deque, List, Optional

import einops
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from mindmap.cli.args import ClosedLoopAppArgs
from mindmap.closed_loop.policies.policy_base import PolicyBase
from mindmap.data_loading.data_types import (
    includes_nvblox,
    includes_pcd,
    includes_policy_states,
    includes_rgb,
)
from mindmap.embodiments.arm.embodiment import ArmEmbodiment
from mindmap.embodiments.embodiment_base import EmbodimentBase
from mindmap.embodiments.observation_base import ObservationBase
from mindmap.embodiments.state_base import PolicyStateBase, state_tensor_from_history
from mindmap.image_processing.image_conversions import convert_rgb_to_model_input
from mindmap.mapping.isaaclab_nvblox_mapper import IsaacLabNvbloxMapper
from mindmap.mapping.nvblox_mapper_constants import MAPPER_TO_ID
from mindmap.model_utils.checkpoint import get_model, load_inference_checkpoint
from mindmap.model_utils.task_to_predict_head_yaw import get_predict_head_yaw_from_task
from mindmap.visualization.visualizer import Visualizer


class NvbloxDiffuserActorPolicy(PolicyBase):
    """A policy which implements nvblox DiffuserActor to generate goals."""

    def __init__(self, args: ClosedLoopAppArgs, device: str):
        self.args = args
        self.device = device

        # Model
        self.model = self._load_model()
        self.predict_head_yaw = get_predict_head_yaw_from_task(self.args.task)

        # Mapper
        self.isaaclab_nvblox_mapper = None
        if includes_nvblox(args.data_type):
            self.isaaclab_nvblox_mapper = IsaacLabNvbloxMapper(args.data_type, args, device)

        # History of past states
        self.policy_state_history_deque: Deque[Optional[PolicyStateBase]] = deque(
            [None] * args.num_history, maxlen=args.num_history
        )

        # Visualizer
        self.visualizer = None
        if self.args.visualize:
            self.visualizer = Visualizer(args)

        # Reset the policy for good measure.
        self.reset()

    def reset(self):
        """Resets the policy's internal state."""
        self.policy_state_history_deque = deque(
            [None] * self.args.num_history, maxlen=self.args.num_history
        )
        if includes_nvblox(self.args.data_type):
            print("Clearing nvblox mapper.")
            self.isaaclab_nvblox_mapper.clear()

    def step(self, current_state: PolicyStateBase, observation: ObservationBase) -> None:
        """Called every simulation step to update policy's internal state."""
        # Update the reconstruction.
        if includes_nvblox(self.args.data_type):
            self.isaaclab_nvblox_mapper.decay()
            for camera_handler in observation.get_cameras().values():
                self.isaaclab_nvblox_mapper.update_reconstruction_from_camera(camera_handler)

    def get_new_goal(
        self,
        embodiment: EmbodimentBase,
        current_state: PolicyStateBase,
        observation: ObservationBase,
    ) -> List[PolicyStateBase]:
        """Generates a goal given the current state and camera observations."""
        # Update the past trajectory.
        self._update_gripper_history(current_state)
        # Extract the model inputs from IsaacLab.
        # TODO(Vik): This does not support multiple embodiments. Fix this in separate MR
        model_inputs = self._get_model_inputs(embodiment=embodiment, observation=observation)
        # Run the model
        pred, head_yaw_pred, _, encoded_inputs, cross_attn_weights = self.model(
            gt_gripper_pred=None,
            gt_head_yaw=None,
            rgb_obs=model_inputs["rgb_obs"],
            pcd_obs=model_inputs["pcd_obs"],
            pcd_valid_mask=model_inputs["pcd_valid_mask"],
            vertex_features=model_inputs["vertex_features"],
            vertices=model_inputs["vertices"],
            vertices_valid_mask=model_inputs["vertices_valid_mask"],
            instruction=None,
            gripper_history=model_inputs["gripper_history"],
            run_inference=True,
        )
        # Get number of grippers
        num_grippers = embodiment.get_num_grippers()
        num_items_in_gripper_prediction = embodiment.get_number_of_items_in_gripper_prediction()[1]
        assert pred.shape == (
            1,
            self.args.prediction_horizon,
            num_grippers,
            num_items_in_gripper_prediction,
        )

        # Join the gripper dimension with the states
        pred = einops.rearrange(pred, "b l ngripper c -> b l (ngripper c)")

        policy_state_pred_tensor = embodiment.get_policy_state_tensor_from_model_prediction(
            pred, head_yaw_pred
        )
        pred_states = embodiment.policy_state_type.history_from_tensor(policy_state_pred_tensor)

        # Visualize
        if self.args.visualize:
            self._visualize(model_inputs, encoded_inputs, pred, cross_attn_weights)
        # Extract the goal from the predicted states
        if self.args.use_keyposes:
            # Return the next n predicted keyposes
            return pred_states[: self.args.prediction_horizon]
        else:
            # Return the last of the predicted trajectory states
            return [pred_states[-1]]

    def _get_model_inputs(self, embodiment: EmbodimentBase, observation: ObservationBase):
        """
        Extracts the model inputs from IsaacLab.
        """
        # Initialize the observations to None
        samples = {
            "pcd_obs": None,
            "pcd_valid_mask": None,
            "rgb_obs": None,
            "vertex_features": None,
            "vertices": None,
            "vertices_valid_mask": None,
            "gripper_history": None,
        }

        if includes_policy_states(self.args.data_type):
            policy_state_history_tensor = state_tensor_from_history(self.policy_state_history_deque)
            samples["gripper_history"] = embodiment.policy_state_type.split_gripper_tensor(
                policy_state_history_tensor
            )

        batch_size = 1

        # Construct the required shapes.
        cams = observation.get_cameras()
        num_cams = len(cams)
        expected_num_cams = 2 if self.args.add_external_cam else 1
        assert num_cams == expected_num_cams

        # Confirm identical image size for every camera
        sizes = {cam.get_image_size() for cam in cams.values()}
        assert len(sizes) == 1, "all cameras must share the same resolution"
        image_size = sizes.pop()

        if includes_rgb(self.args.data_type):
            rgb_stack = [
                convert_rgb_to_model_input(cam.get_rgb().squeeze(0)).unsqueeze(0)
                for cam in cams.values()
            ]
            samples["rgb_obs"] = torch.stack(rgb_stack, dim=1)
            assert samples["rgb_obs"].shape == (
                batch_size,
                num_cams,
                3,
                image_size[0],
                image_size[1],
            )

        if includes_pcd(self.args.data_type):
            pcd_stack = [cam.get_pcd().unsqueeze(0) for cam in cams.values()]
            samples["pcd_obs"] = torch.stack(pcd_stack, dim=1)

            valid_depth_mask_stack = [
                cam.get_valid_depth_mask(self.args.rgbd_min_depth_threshold).unsqueeze(0)
                for cam in cams.values()
            ]
            samples["pcd_valid_mask"] = torch.stack(valid_depth_mask_stack, dim=1)

            assert samples["pcd_obs"].shape == (
                batch_size,
                num_cams,
                3,
                image_size[0],
                image_size[1],
            )

        if includes_nvblox(self.args.data_type):
            samples.update(
                self.isaaclab_nvblox_mapper.get_nvblox_model_inputs(
                    mapper_id=MAPPER_TO_ID.STATIC, remove_zero_features=True
                )
            )

        return samples

    def _visualize(self, model_inputs, encoded_inputs, pred, cross_attn_weights):
        assert self.visualizer is not None
        sample = {
            "rgbs": model_inputs["rgb_obs"],
            "pcds": model_inputs["pcd_obs"],
            "vertex_features": model_inputs["vertex_features"],
            "vertices": model_inputs["vertices"],
            "gripper_history": model_inputs["gripper_history"],
            "is_keypose": torch.tensor(self.args.use_keyposes).reshape((1, 1)),
            "add_external_cam": self.args.add_external_cam,
            "cross_attn_weights": cross_attn_weights,
        }
        sample.update(encoded_inputs)
        self.visualizer.visualize(
            sample,
            self.args.data_type,
            pred,
            isaac_lab_nvblox_mapper=self.isaaclab_nvblox_mapper,
        )
        if not self.args.disable_visualizer_wait_on_key:
            self.visualizer.run_until_space_pressed()

    def _load_model(self):
        # Loading the DiffuserActor checkpoint to run closed loop with.
        model = get_model(self.args).to(self.device)
        model = DDP(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        model = load_inference_checkpoint(self.args.checkpoint, model, device=self.device)
        return model

    def _update_gripper_history(self, current_state: PolicyStateBase):
        # NOTE: For DiffuserActor it expects the current pose as the last pose in
        # the past trajectory. So it should be appended before inference.
        # If this is the first time, fill the history with the current state.
        current_state_copy = copy.deepcopy(current_state)
        if None in self.policy_state_history_deque:
            self.policy_state_history_deque = deque(
                [current_state_copy] * self.args.num_history, maxlen=self.args.num_history
            )
        else:
            self.policy_state_history_deque.append(current_state_copy)

    def run(self):
        pass

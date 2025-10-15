# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import List, Optional, Tuple

import gymnasium as gym
import torch

from mindmap.cli.args import ClosedLoopAppArgs
from mindmap.closed_loop.goals import get_error_to_goal
from mindmap.embodiments.arm.action import ArmEmbodimentAction
from mindmap.embodiments.arm.constants import (
    ARM_GO_TO_NEXT_GOAL_THRESHOLD_DEG,
    ARM_GO_TO_NEXT_GOAL_THRESHOLD_GRIPPER_DIFF,
    ARM_GO_TO_NEXT_GOAL_THRESHOLD_M,
)
from mindmap.embodiments.arm.controller import ArmEmbodimentController
from mindmap.embodiments.arm.estimator import (
    ArmEmbodimentOfflineEstimator,
    ArmEmbodimentOnlineEstimator,
)
from mindmap.embodiments.arm.keypose_estimation import ArmEmbodimentKeyposeEstimator
from mindmap.embodiments.arm.observation import (
    ArmEmbodimentObservation,
    get_camera_item_names_by_encoding_method,
)
from mindmap.embodiments.arm.policy_state import ArmEmbodimentPolicyState
from mindmap.embodiments.arm.robot_state import ArmEmbodimentRobotState
from mindmap.embodiments.embodiment_base import EmbodimentBase, EmbodimentType
from mindmap.isaaclab_utils.isaaclab_camera_handler import IsaacLabCameraHandler


class ArmEmbodiment(EmbodimentBase):
    embodiment_type = EmbodimentType.ARM
    robot_state_type = ArmEmbodimentRobotState
    policy_state_type = ArmEmbodimentPolicyState
    action_type = ArmEmbodimentAction
    controller_type = ArmEmbodimentController
    online_estimator_type = ArmEmbodimentOnlineEstimator
    offline_estimator_type = ArmEmbodimentOfflineEstimator
    observation_type = ArmEmbodimentObservation
    keypose_estimator_type = ArmEmbodimentKeyposeEstimator

    # Make args optional
    def __init__(self, args: ClosedLoopAppArgs = None, device: str = "cuda"):
        super().__init__(device=device)
        self.args = args
        self.wrist_camera_handler = None
        self.table_camera_handler = None
        self.visualization_markers = None
        self.goal_markers = None

    def get_robot_state(self, env: gym.wrappers.common.OrderEnforcing) -> ArmEmbodimentRobotState:
        """Get the current state of the end effector from the environment.

        Args:
            env (gym.wrappers.common.OrderEnforcing): The environment instance.

        Returns:
            ArmEmbodimentRobotState: A class containing:
                - End-effector position (3)
                - End-effector quaternion orientation (4)
                - Gripper jaw positions (2)
        """
        ee_frame = env.unwrapped.scene["ee_frame"]
        assert ee_frame.data.target_frame_names[0] == "end_effector"
        assert (
            ee_frame.data.target_pos_w.shape[0] == 1
        ), "ArmEmbodiment only supports single robot instances"
        ee_pos = ee_frame.data.target_pos_w[0, 0, :].to(self.device)
        ee_quat = ee_frame.data.target_quat_w[0, 0, :].to(self.device)
        robot = env.unwrapped.scene["robot"]
        gripper_pos_xy = robot.data.joint_pos[0, -2:].to(self.device)
        # Return the end effector state as a class.
        return ArmEmbodimentRobotState(
            W_t_W_Eef=ee_pos, q_wxyz_W_Eef=ee_quat, gripper_jaw_positions=gripper_pos_xy
        )

    def initialize_camera_handlers(self, env: gym.wrappers.common.OrderEnforcing):
        self.wrist_camera = env.unwrapped.scene["wrist_cam"]
        self.wrist_camera_handler = IsaacLabCameraHandler(self.wrist_camera, "wrist")
        self.camera_handlers = [self.wrist_camera_handler]
        if self.args.add_external_cam:
            self.table_camera = env.unwrapped.scene["table_cam"]
            self.table_camera_handler = IsaacLabCameraHandler(self.table_camera, "table")
            self.camera_handlers.append(self.table_camera_handler)

    def get_observation(self, env: gym.wrappers.common.OrderEnforcing) -> ArmEmbodimentObservation:
        """Get the current observation of the environment.

        Returns:
            ArmEmbodimentObservation: A class containing:
                - Table camera
                - Wrist camera
        """
        if self.wrist_camera_handler is None:
            self.initialize_camera_handlers(env)

        return ArmEmbodimentObservation(
            table_camera=self.table_camera_handler, wrist_camera=self.wrist_camera_handler
        )

    def is_goal_reached(
        self,
        current_state: ArmEmbodimentPolicyState,
        goal_state: ArmEmbodimentPolicyState,
        print_errors: bool = False,
    ) -> bool:
        error_m, error_deg = get_error_to_goal(
            W_t_W_Eef=current_state.W_t_W_Eef,
            q_W_Eef=current_state.q_wxyz_W_Eef,
            W_t_W_Goal=goal_state.W_t_W_Eef,
            q_W_Goal=goal_state.q_wxyz_W_Eef,
        )
        gripper_diff = torch.abs(goal_state.gripper_closedness - current_state.gripper_closedness)
        if print_errors:
            print(
                f"Errors to goals: {error_m:.3f} m, {error_deg:.1f} deg, {gripper_diff.item():.1f} openness"
            )
        return (
            error_m < ARM_GO_TO_NEXT_GOAL_THRESHOLD_M
            and error_deg < ARM_GO_TO_NEXT_GOAL_THRESHOLD_DEG
            and gripper_diff < ARM_GO_TO_NEXT_GOAL_THRESHOLD_GRIPPER_DIFF
        )

    def add_intermediate_goals(
        self,
        current_state: ArmEmbodimentPolicyState,
        goal_state: List[ArmEmbodimentPolicyState],
    ) -> Tuple[List[ArmEmbodimentPolicyState], List[bool]]:
        # For the robot arm, we don't add intermediate goals.
        # NOTE(remos): This could be enabled if we experience issues with set point jumps for the robot arm.
        assert self.args.max_intermediate_distance_m is None
        return goal_state, [False]

    def visualize_robot_state(
        self,
        robot_state: ArmEmbodimentRobotState,
        goal_state: Optional[ArmEmbodimentPolicyState] = None,
    ):
        from mindmap.visualization.isaac_lab_visualization import get_axis_markers

        # Create the marker if they don't yet exist.
        if self.visualization_markers is None:
            self.visualization_markers = get_axis_markers(
                marker_names=["eef"], prim_path="/Visuals/eef_frames"
            )
        if goal_state is not None and self.goal_markers is None:
            self.goal_markers = get_axis_markers(
                marker_names=["eef_goal"], prim_path="/Visuals/goal_markers"
            )
        # Update positions and orientations.
        translations = torch.stack([robot_state.W_t_W_Eef], dim=0)
        orientations = torch.stack([robot_state.q_wxyz_W_Eef], dim=0)
        self.visualization_markers.visualize(translations=translations, orientations=orientations)
        if goal_state is not None:
            translations_goal = torch.stack([goal_state.W_t_W_Eef], dim=0)
            orientations_goal = torch.stack([goal_state.q_wxyz_W_Eef], dim=0)
            self.goal_markers.visualize(
                translations=translations_goal, orientations=orientations_goal
            )

    def get_policy_state_tensor_from_model_prediction(
        self, trajectory_pred: torch.Tensor, head_yaw_pred: torch.Tensor
    ) -> torch.Tensor:
        """Converts a model prediction to a policy state tensor."""
        # Just ignore the head yaw prediction for the arm embodiment.
        return trajectory_pred

    def get_camera_item_names_by_encoding_method(self, add_external_cam: bool):
        return get_camera_item_names_by_encoding_method(add_external_cam=add_external_cam)

    def get_number_of_items_in_gripper_prediction(self):
        """Returns the number of items in the gripper prediction. This is always number of [grippers, number of prediction outputs]"""
        return [1, 8]

    def get_num_grippers(self):
        """Returns the number of grippers. This is always 1 for the arm embodiment."""
        return 1

    def convert_action_to_tensor(self, action: ArmEmbodimentAction) -> torch.Tensor:
        """Converts an action to a tensor."""
        return action.to_tensor()

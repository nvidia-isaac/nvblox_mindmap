# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from scipy.spatial.transform import Rotation as R, Slerp
import torch

from mindmap.cli.args import ClosedLoopAppArgs
from mindmap.closed_loop.goals import get_error_to_goal
from mindmap.closed_loop.intermediate_goals import TASK_TYPE_TO_MAX_INTERMEDIATE_DISTANCE_M
from mindmap.embodiments.embodiment_base import EmbodimentBase, EmbodimentType
from mindmap.embodiments.humanoid.action import HumanoidEmbodimentAction
from mindmap.embodiments.humanoid.constants import (
    HUMANOID_GO_TO_NEXT_GOAL_THRESHOLD_DEG,
    HUMANOID_GO_TO_NEXT_GOAL_THRESHOLD_GRIPPER_DIFF,
    HUMANOID_GO_TO_NEXT_GOAL_THRESHOLD_HEAD_YAW_DEG,
    HUMANOID_GO_TO_NEXT_GOAL_THRESHOLD_M,
)
from mindmap.embodiments.humanoid.controller import HumanoidEmbodimentController
from mindmap.embodiments.humanoid.estimator import (
    HumanoidEmbodimentOfflineEstimator,
    HumanoidEmbodimentOnlineEstimator,
)
from mindmap.embodiments.humanoid.joint_indices import HumanoidJointIndices
from mindmap.embodiments.humanoid.keypose_estimation import HumanoidEmbodimentKeyposeEstimator
from mindmap.embodiments.humanoid.observation import (
    HumanoidEmbodimentObservation,
    get_camera_item_names_by_encoding_method,
)
from mindmap.embodiments.humanoid.policy_state import HumanoidEmbodimentPolicyState
from mindmap.embodiments.humanoid.robot_state import HumanoidEmbodimentRobotState
from mindmap.isaaclab_utils.isaaclab_camera_handler import IsaacLabCameraHandler
from mindmap.model_utils.task_to_predict_head_yaw import get_predict_head_yaw_from_task
from mindmap.tasks.tasks import Tasks


class HumanoidEmbodiment(EmbodimentBase):
    embodiment_type = EmbodimentType.HUMANOID
    robot_state_type = HumanoidEmbodimentRobotState
    policy_state_type = HumanoidEmbodimentPolicyState
    action_type = HumanoidEmbodimentAction
    controller_type = HumanoidEmbodimentController
    online_estimator_type = HumanoidEmbodimentOnlineEstimator
    offline_estimator_type = HumanoidEmbodimentOfflineEstimator
    observation_type = HumanoidEmbodimentObservation
    keypose_estimator_type = HumanoidEmbodimentKeyposeEstimator

    def __init__(self, task: Tasks = None, args: ClosedLoopAppArgs = None, device: str = "cuda"):
        super().__init__(device=device)
        self.args = args
        self.pov_camera_handler = None
        self.external_camera_handler = None
        self.visualization_markers = None
        self.goal_markers = None
        # Initialize the hand joint name to index map.
        self.left_hand_joint_indices = list(HumanoidJointIndices.left_hand_name_to_idx_map.values())
        self.right_hand_joint_indices = list(
            HumanoidJointIndices.right_hand_name_to_idx_map.values()
        )
        self.predict_head_yaw = get_predict_head_yaw_from_task(task) if task is not None else None
        if (
            hasattr(self.args, "max_intermediate_distance_m")
            and self.args.max_intermediate_distance_m is None
        ):
            self.args.max_intermediate_distance_m = TASK_TYPE_TO_MAX_INTERMEDIATE_DISTANCE_M[
                task.name
            ]

    def get_robot_state(
        self, env: gym.wrappers.common.OrderEnforcing
    ) -> HumanoidEmbodimentRobotState:
        """Get the current state of the end effector from the environment.

        Args:
            env (gym.wrappers.common.OrderEnforcing): The environment instance.

        Returns:
            HumanoidEmbodimentRobotState: A class containing:
                - Left end-effector position (3)
                - Left end-effector quaternion orientation (4)
                - Left hand joint states (11)
                - Right end-effector position (3)
                - Right end-effector quaternion orientation (4)
                - Right hand joint states (11)
        """
        from isaaclab_tasks.manager_based.manipulation.pick_place.mdp.observations import (
            get_hand_state,
            get_head_state,
            get_left_eef_pos,
            get_left_eef_quat,
            get_right_eef_pos,
            get_right_eef_quat,
        )

        # Check there's only one robot instance.
        body_pos_w = env.unwrapped.scene["robot"].data.body_pos_w
        assert body_pos_w.shape[0] == 1, "HumanoidEmbodiment only supports single robot instances"
        # Get the end effector positions and orientations.
        left_eef_pos = get_left_eef_pos(env.unwrapped).squeeze().to(self.device)
        left_eef_quat = get_left_eef_quat(env.unwrapped).squeeze().to(self.device)
        right_eef_pos = get_right_eef_pos(env.unwrapped).squeeze().to(self.device)
        right_eef_quat = get_right_eef_quat(env.unwrapped).squeeze().to(self.device)
        head_yaw_rad = get_head_state(env.unwrapped).squeeze()[2:3].to(self.device)
        # Get the hand joint states, and then split them into left and right.
        hand_joint_states = get_hand_state(env.unwrapped).to(self.device)
        hand_joint_states_left = hand_joint_states[:, self.left_hand_joint_indices].squeeze()
        hand_joint_states_right = hand_joint_states[:, self.right_hand_joint_indices].squeeze()
        # Return the end effector state as a class.
        return HumanoidEmbodimentRobotState(
            W_t_W_LeftEef=left_eef_pos,
            q_wxyz_W_LeftEef=left_eef_quat,
            left_hand_joint_states=hand_joint_states_left,
            W_t_W_RightEef=right_eef_pos,
            q_wxyz_W_RightEef=right_eef_quat,
            right_hand_joint_states=hand_joint_states_right,
            head_yaw_rad=head_yaw_rad,
        )

    def visualize_robot_state(
        self,
        robot_state: HumanoidEmbodimentRobotState,
        goal_state: Optional[HumanoidEmbodimentPolicyState] = None,
    ):
        from mindmap.visualization.isaac_lab_visualization import get_axis_markers

        # Create the marker if they don't yet exist.
        if self.visualization_markers is None:
            self.visualization_markers = get_axis_markers(
                marker_names=["left_eef", "right_eef"], prim_path="/Visuals/eef_frames"
            )
        if goal_state is not None and self.goal_markers is None:
            self.goal_markers = get_axis_markers(
                marker_names=["left_eef_goal", "right_eef_goal"], prim_path="/Visuals/goal_markers"
            )
        # Update positions and orientations.
        translations = torch.stack([robot_state.W_t_W_LeftEef, robot_state.W_t_W_RightEef], dim=0)
        orientations = torch.stack(
            [robot_state.q_wxyz_W_LeftEef, robot_state.q_wxyz_W_RightEef], dim=0
        )
        self.visualization_markers.visualize(translations=translations, orientations=orientations)
        if goal_state is not None:
            translations_goal = torch.stack(
                [goal_state.W_t_W_LeftEef, goal_state.W_t_W_RightEef], dim=0
            )
            orientations_goal = torch.stack(
                [goal_state.q_wxyz_W_LeftEef, goal_state.q_wxyz_W_RightEef], dim=0
            )
            self.goal_markers.visualize(
                translations=translations_goal, orientations=orientations_goal
            )

    def get_policy_state_tensor_from_model_prediction(
        self, trajectory_pred: torch.Tensor, head_yaw_pred: torch.Tensor
    ) -> torch.Tensor:
        """Converts a model prediction to a policy state tensor."""
        # Concatenate the gripper and head yaw predictions to get the policy state prediction
        if not self.predict_head_yaw:
            # Add zeros for the head yaw prediction if we don't predict it.
            batch_size = trajectory_pred.shape[0]
            assert trajectory_pred.shape[1] == self.args.prediction_horizon
            head_yaw_pred = torch.zeros(
                (batch_size, self.args.prediction_horizon, 1), device=self.device
            )
        assert trajectory_pred.shape[:-1] == head_yaw_pred.shape[:-1]
        return torch.cat([trajectory_pred, head_yaw_pred], dim=-1)

    def initialize_camera_handlers(self, env: gym.wrappers.common.OrderEnforcing):
        self.pov_camera = env.unwrapped.scene["robot_pov_cam"]
        self.pov_camera_handler = IsaacLabCameraHandler(self.pov_camera, "pov")
        self.camera_handlers = [self.pov_camera_handler]
        if self.args.add_external_cam:
            self.external_camera = env.unwrapped.scene["external_cam"]
            self.external_camera_handler = IsaacLabCameraHandler(self.external_camera, "external")
            self.camera_handlers.append(self.external_camera_handler)

    def get_observation(
        self, env: gym.wrappers.common.OrderEnforcing
    ) -> HumanoidEmbodimentObservation:
        """Get the current observation of the environment.

        Returns:
            HumanoidEmbodimentObservation: A class containing:
                - Head camera
        """
        if self.pov_camera_handler is None:
            self.initialize_camera_handlers(env)
        return HumanoidEmbodimentObservation(
            pov_camera=self.pov_camera_handler, external_camera=self.external_camera_handler
        )

    @staticmethod
    def interpolate_quaternion(q0, q1, t):
        """
        Performs spherical linear interpolation (SLERP) between two quaternions.

        Args:
            q0 (torch.Tensor): The starting quaternion in (w, x, y, z) format.
            q1 (torch.Tensor): The ending quaternion in (w, x, y, z) format.
            t (float): Interpolation parameter between 0.0 (q0) and 1.0 (q1).

        Returns:
            torch.Tensor: The interpolated quaternion in (w, x, y, z) format.
        """
        # Prepare key times and rotations for Slerp
        key_times = [0, 1]
        # Convert quaternions from (w, x, y, z) to (x, y, z, w) for scipy
        q0_cpu = q0.cpu().numpy()
        q1_cpu = q1.cpu().numpy()
        q0_xyzw = [q0_cpu[1], q0_cpu[2], q0_cpu[3], q0_cpu[0]]
        q1_xyzw = [q1_cpu[1], q1_cpu[2], q1_cpu[3], q1_cpu[0]]
        key_rots = R.from_quat([q0_xyzw, q1_xyzw])
        slerp = Slerp(key_times, key_rots)
        intermediate_rot = slerp([t])[0]
        # Convert back to (w, x, y, z)
        intermediate_quat_xyzw = intermediate_rot.as_quat()
        intermediate_quat_wxyz = torch.tensor(
            [
                intermediate_quat_xyzw[3],
                intermediate_quat_xyzw[0],
                intermediate_quat_xyzw[1],
                intermediate_quat_xyzw[2],
            ],
            dtype=q0.dtype,
            device=q0.device,
        )
        return intermediate_quat_wxyz

    def add_intermediate_goals(
        self,
        current_state: HumanoidEmbodimentPolicyState,
        initial_goal_state_list: List[HumanoidEmbodimentPolicyState],
    ) -> Tuple[List[HumanoidEmbodimentPolicyState], List[bool]]:
        """
        Add intermediate goals to the initial goal state list.
        """
        updated_goal_state_list = []
        is_intermediate_goal_list = []
        for goal_state in initial_goal_state_list:
            # For each goal state, add intermediate goals if needed.
            goals, is_intermediate_goal = self.prepend_intermediate_goals(current_state, goal_state)
            updated_goal_state_list.extend(goals)
            is_intermediate_goal_list.extend(is_intermediate_goal)
        return updated_goal_state_list, is_intermediate_goal_list

    def prepend_intermediate_goals(
        self,
        current_state: HumanoidEmbodimentPolicyState,
        goal_state: HumanoidEmbodimentPolicyState,
    ) -> Tuple[List[HumanoidEmbodimentPolicyState], List[bool]]:
        """
        Computes and returns a list of intermediate policy states (goals) between the current and goal states,
        interpolating positions, orientations, and head yaw as needed.
        Intermediate goals are inserted for sections that are larger than the maximum allowed intermediate distance (max_intermediate_distance_m).

        Returns:
            Tuple[List[HumanoidEmbodimentPolicyState], List[bool]]:
                - A list of intermediate and final goal policy states.
                - A list of booleans indicating whether each state is an intermediate goal (True) or the final goal (False).
        """
        if goal_state is None:
            return [goal_state], [False]
        # Get the distance between the current and goal states.
        current_to_goal_vector_left = goal_state.W_t_W_LeftEef - current_state.W_t_W_LeftEef
        distance_left = torch.norm(current_to_goal_vector_left)
        current_to_goal_vector_right = goal_state.W_t_W_RightEef - current_state.W_t_W_RightEef
        distance_right = torch.norm(current_to_goal_vector_right)
        head_yaw_diff = goal_state.head_yaw_rad - current_state.head_yaw_rad
        distance = max(distance_left, distance_right)

        # Add intermediate goals if needed.
        goals = []
        is_intermediate_goal = []
        if (
            self.args.max_intermediate_distance_m is not None
            and distance > self.args.max_intermediate_distance_m
        ):
            num_intermediate_goals = math.floor(distance / self.args.max_intermediate_distance_m)
            steps_to_goal = num_intermediate_goals + 1

            for idx in range(num_intermediate_goals):
                t = (idx + 1) / steps_to_goal

                # Interpolate position.
                intermediate_goal_position_left = (
                    current_to_goal_vector_left * t + current_state.W_t_W_LeftEef
                )
                intermediate_goal_position_right = (
                    current_to_goal_vector_right * t + current_state.W_t_W_RightEef
                )

                # Interpolate orientation.
                intermediate_quat_wxyz_left = self.interpolate_quaternion(
                    current_state.q_wxyz_W_LeftEef, goal_state.q_wxyz_W_LeftEef, t
                )
                intermediate_quat_wxyz_right = self.interpolate_quaternion(
                    current_state.q_wxyz_W_RightEef, goal_state.q_wxyz_W_RightEef, t
                )
                intermediate_yaw_rad = head_yaw_diff * t + current_state.head_yaw_rad

                # Add the intermediate goal as a policy state.
                goals.append(
                    HumanoidEmbodimentPolicyState(
                        W_t_W_LeftEef=intermediate_goal_position_left,
                        q_wxyz_W_LeftEef=intermediate_quat_wxyz_left,
                        left_hand_closedness=current_state.left_hand_closedness,
                        W_t_W_RightEef=intermediate_goal_position_right,
                        q_wxyz_W_RightEef=intermediate_quat_wxyz_right,
                        right_hand_closedness=current_state.right_hand_closedness,
                        head_yaw_rad=intermediate_yaw_rad,
                    )
                )
                is_intermediate_goal.append(True)

            # Add the final goal as a policy state.
            goals.append(goal_state)
            is_intermediate_goal.append(False)
            return goals, is_intermediate_goal
        else:
            return [goal_state], [False]

    def are_errors_to_goal_within_threshold(
        self, error_m: float, error_deg: float, gripper_diff: float, is_intermediate_goal: bool
    ) -> bool:
        """
        Checks if the errors to the goal are within the threshold.
        If the goal is an intermediate goal, the threshold is half of the maximum intermediate distance.
        """
        if is_intermediate_goal:
            assert self.args.max_intermediate_distance_m is not None
            # In case of intermediate goal, we don't want to constraint too much.
            # Intermediate goals are only to avoid big set point jumps.
            return error_m < self.args.max_intermediate_distance_m * 0.5
        else:
            return (
                error_m < HUMANOID_GO_TO_NEXT_GOAL_THRESHOLD_M
                and error_deg < HUMANOID_GO_TO_NEXT_GOAL_THRESHOLD_DEG
                and gripper_diff < HUMANOID_GO_TO_NEXT_GOAL_THRESHOLD_GRIPPER_DIFF
            )

    def is_goal_reached(
        self,
        current_state: HumanoidEmbodimentPolicyState,
        goal_state: HumanoidEmbodimentPolicyState,
        is_intermediate_goal: bool = False,
        print_errors: bool = False,
    ) -> bool:
        error_left_m, error_left_deg = get_error_to_goal(
            W_t_W_Eef=current_state.W_t_W_LeftEef,
            q_W_Eef=current_state.q_wxyz_W_LeftEef,
            W_t_W_Goal=goal_state.W_t_W_LeftEef,
            q_W_Goal=goal_state.q_wxyz_W_LeftEef,
        )
        error_right_m, error_right_deg = get_error_to_goal(
            W_t_W_Eef=current_state.W_t_W_RightEef,
            q_W_Eef=current_state.q_wxyz_W_RightEef,
            W_t_W_Goal=goal_state.W_t_W_RightEef,
            q_W_Goal=goal_state.q_wxyz_W_RightEef,
        )
        gripper_left_diff = torch.abs(
            goal_state.left_hand_closedness - current_state.left_hand_closedness
        )
        gripper_right_diff = torch.abs(
            goal_state.right_hand_closedness - current_state.right_hand_closedness
        )
        error_m = max(error_left_m, error_right_m)
        error_deg = max(error_left_deg, error_right_deg)
        gripper_diff = max(gripper_left_diff, gripper_right_diff)
        if print_errors:
            print(
                f"Left Eef errors to goal: {error_left_m:.3f} m, {error_left_deg:.1f} deg, {gripper_left_diff.item():.1f} openness"
            )
            print(
                f"Right Eef errors to goal: {error_right_m:.3f} m, {error_right_deg:.1f} deg, {gripper_right_diff.item():.1f} openness"
            )
        is_goal_reached = self.are_errors_to_goal_within_threshold(
            error_m, error_deg, gripper_diff, is_intermediate_goal
        )
        if self.predict_head_yaw:
            is_goal_reached = is_goal_reached and self.is_head_yaw_goal_reached(
                current_state, goal_state, print_errors
            )
        return is_goal_reached

    def is_head_yaw_goal_reached(
        self,
        robot_state: HumanoidEmbodimentPolicyState,
        goal_state: HumanoidEmbodimentPolicyState,
        print_errors: bool = False,
    ) -> bool:
        # NOTE(remos): This function does not handle wrap arounds,
        #              but we don't expect any because of the limits on head movement.
        head_yaw_diff = math.degrees(
            torch.abs(goal_state.head_yaw_rad - robot_state.head_yaw_rad).item()
        )
        if print_errors:
            print(f"Head yaw error: {head_yaw_diff:.1f} deg")
        return head_yaw_diff < HUMANOID_GO_TO_NEXT_GOAL_THRESHOLD_HEAD_YAW_DEG

    def get_camera_item_names_by_encoding_method(self, add_external_cam: bool):
        return get_camera_item_names_by_encoding_method(add_external_cam)

    def get_number_of_items_in_gripper_prediction(self):
        """Returns the number of items in the gripper prediction. This is always number of [grippers, number of prediction outputs]"""
        return [2, 8]

    def get_num_grippers(self):
        """Returns the number of grippers. This is always 2 for the humanoid embodiment."""
        return 2

    def convert_action_to_tensor(self, action: HumanoidEmbodimentAction) -> torch.Tensor:
        """Converts an action to a tensor."""
        return action.to_tensor(include_head_yaw=self.predict_head_yaw)


def map_to_humanoid_action_space(
    task: Tasks,
    left_wrist_pose: torch.Tensor,
    right_wrist_pose: torch.Tensor,
    head_yaw_rad: torch.Tensor,
    hand_joints: torch.Tensor,
) -> torch.Tensor:
    """
    Returns an action compatible with the humanoid embodiment action space from the input tensors.
    NOTE: Main purpose of this function is to convert unconstrained hand joints angles
          into the action space which only allows one definition for open and closed hands.

    Args:
        left_wrist_pose (torch.Tensor): The pose of the left wrist, shape (7,).
        right_wrist_pose (torch.Tensor): The pose of the right wrist, shape (7,).
        head_yaw_rad (torch.Tensor): The yaw angle of the head in radians, shape (1,).
        hand_joints (torch.Tensor): The combined tensor containing joint values for both left and right hands, shape (22,).

    Returns:
        torch.Tensor: The action inferred from the input.
    """
    # Construct the robot state tensor from the input tensors.
    left_hand_joints = hand_joints[
        HumanoidJointIndices.left_joints_in_combined_hands_tensor_indices
    ]
    right_hand_joints = hand_joints[
        HumanoidJointIndices.right_joints_in_combined_hands_tensor_indices
    ]
    robot_state_tensor = torch.concatenate(
        [
            left_wrist_pose,
            left_hand_joints,
            right_wrist_pose,
            right_hand_joints,
            head_yaw_rad,
        ]
    )

    # Convert from RobotState to PolicyState to Action
    # This ensures that the returned action is compatible with the action space.
    embodiment = HumanoidEmbodiment(task)
    robot_state = HumanoidEmbodimentRobotState.from_tensor(robot_state_tensor)
    policy_state = embodiment.get_policy_state_from_embodiment_state(
        robot_state, last_goal_state=None
    )
    action = embodiment.get_action_from_policy_state(policy_state)
    return embodiment.convert_action_to_tensor(action)

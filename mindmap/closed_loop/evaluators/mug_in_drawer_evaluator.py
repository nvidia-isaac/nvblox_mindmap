# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import os
from typing import Dict

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import wandb

from mindmap.closed_loop.evaluators.arm_evaluator import ArmEvaluatorBase
from mindmap.embodiments.arm.robot_state import ArmEmbodimentRobotState
from mindmap.tasks.tasks import Tasks


class MugInDrawerEvaluator(ArmEvaluatorBase):
    """
    The purpose of this class is to evaluate the performance of the mug in drawer task
    by tracking it the mug was lifted, moved, and if it was released in the drawer.
    """

    def __init__(
        self,
        task: Tasks,
        eval_file_path: os.path = None,
        wandb_step_id: int = None,
        gripper_closedness_threshold: float = 0.5,
        checkpoint_name=None,
    ) -> None:
        """
        Initializes the MugInDrawerEvaluator object.

        Args:
            eval_file_path (os.path, optional): The path where the the evaluation file should be stored.
                                                Defaults to None (which means no evaluation file will be stored).
            wandb_step_id (int, optional): The ID of the current step in the W&B run. Defaults to None.
            gripper_closedness_threshold (float, optional): The threshold for the closeness of the gripper. Defaults to 0.5.
            checkpoint_name (str, optional): Name of checkpoint used for evaluation.
        """
        assert task == Tasks.MUG_IN_DRAWER
        super().__init__(task, eval_file_path, wandb_step_id, checkpoint_name)
        self.drawer_size = np.array([0.4, 0.65, 0.1])
        self.mug_radius = 0.05
        self.mug_height = 0.1
        self.gripper_closedness_threshold = gripper_closedness_threshold

        ## Setting up thresholds for evaluation the mug in drawer task:
        # The mug is considered to be moved if it is moved more than its radius in xy.
        self.min_distance_xy_moved_thresh = self.mug_radius
        # The mug is considered to be lifted if it is lifted more than half its height in z.
        self.min_distance_z_lifted_thresh = self.mug_height / 2.0

    def init_demo(self, demo_name: str, retry_idx: int, env: gym.Env) -> None:
        """
        Initializes the demo for the mug in drawer task. Should be called at the beginning of each new demo.

        Args:
            demo_name (str): The name of the demo.
            retry_idx (int): The index of the current retry.
            env (gym.Env): The environment in which the demo is being initialized.
        """
        self.demo_name = demo_name
        self.retry_idx = retry_idx
        # Get the initial positions of the mug.
        self.initial_mug_position = self._get_mug_position(env)
        self.drawer_position = self._get_drawer_position(env, "bottom_of_drawer_with_mugs")
        self.wrong_drawer_position = self._get_drawer_position(env, "bottom_of_drawer_with_boxes")

        # Reset the evaluation variables.
        self.mug_has_been_lifted = False
        self.mug_has_been_moved = False
        self.mug_has_been_in_drawer = False
        self.mug_has_been_in_wrong_drawer = False
        self.mug_has_been_released_in_drawer = False

    def evaluate_step(self, observed_state: ArmEmbodimentRobotState, env: gym.Env) -> None:
        """
        Evaluates the current state of the mug in drawer task.
        Args:
            observed_state (State): The observed state of the environment.
            env (gym.Env): The environment in which the mug in drawer task is being evaluated.

        Returns:
            tuple: A tuple containing:
                - bool: Whether the mug is currently lifted.
                - bool: Whether the mug has been moved from its initial position.
                - bool: Whether the mug is currently in the target drawer.
        """
        # Get the initial mug position
        initial_mug_z = self.initial_mug_position[2]
        initial_mug_xy = self.initial_mug_position[:2]

        # Get the current mug positions
        mug_position = self._get_mug_position(env)
        mug_z = mug_position[2]
        mug_xy = mug_position[:2]

        # Check constant number of mug
        assert len(self.initial_mug_position) == len(mug_position)

        # Check whether the mug have been lifted
        delta_mug_z = mug_z - initial_mug_z
        mug_is_lifted = bool(delta_mug_z > self.min_distance_z_lifted_thresh)
        self.mug_has_been_lifted |= mug_is_lifted

        # Check whether the mug have been moved
        delta_mug_xy = np.linalg.norm(mug_xy - initial_mug_xy)
        mug_is_moved = bool(delta_mug_xy > self.min_distance_xy_moved_thresh)
        self.mug_has_been_moved |= mug_is_moved

        mug_is_in_drawer = self._mug_is_in_drawer(mug_position, self.drawer_position)
        self.mug_has_been_in_drawer |= mug_is_in_drawer

        mug_is_in_wrong_drawer = self._mug_is_in_drawer(mug_position, self.wrong_drawer_position)
        self.mug_has_been_in_wrong_drawer |= mug_is_in_wrong_drawer

        if self._gripper_is_open(observed_state) and mug_is_in_drawer:
            self.mug_has_been_released_in_drawer = True

        return mug_is_lifted, mug_is_moved, mug_is_in_drawer

    def finalize_demo(self, observed_state: ArmEmbodimentRobotState, env) -> None:
        """
        Finalizes the evaluation of a demo for the mug in drawer task.
        Saves the most recent evaluation results to a file.

        Args:
            observed_state (State): The observed state of the environment.
            env (gym.Env): The environment in which the mug in drawer task is being evaluated.
        """
        self.evaluate_step(observed_state, env)
        demo_key = f"{self.demo_name}_{self.retry_idx}"
        success = self.mug_has_been_released_in_drawer
        self.eval_dict[demo_key] = {
            "demo": self.demo_name,
            "success": success,
            "mug_has_been_lifted": self.mug_has_been_lifted,
            "mug_has_been_moved": self.mug_has_been_moved,
            "mug_has_been_in_drawer": self.mug_has_been_in_drawer,
            "mug_has_been_in_wrong_drawer": self.mug_has_been_in_wrong_drawer,
        }

        print(f"Closed loop success of {self.demo_name}: {success}")

        self.maybe_write_eval_file()

    def summarize_demos(self) -> Dict:
        """
        Summarizes the evaluation of the mug in drawer task for multiple demos.
        Saves the evaluation of all demos together with the summary to a file and returns eval dict.
        """
        summary_dict = {
            "demo": {},
            "success": 0,
            "mug_has_been_lifted": 0,
            "mug_has_been_moved": 0,
            "mug_has_been_in_drawer": 0,
            "mug_has_been_in_wrong_drawer": 0,
        }

        for key, demo_eval_dict in self.eval_dict.items():
            if key in ["summary", "metadata"]:
                continue
            for key, value in demo_eval_dict.items():
                if isinstance(value, bool):
                    summary_dict[key] += int(value)
                else:
                    self._count_occurrences(summary_dict[key], value)

        # Compute mean values
        num_demos = sum(summary_dict["demo"].values())
        summary_dict["num_demos"] = num_demos
        summary_dict["success_rate"] = summary_dict["success"] / num_demos
        summary_dict["mug_has_been_lifted_rate"] = summary_dict["mug_has_been_lifted"] / num_demos
        summary_dict["mug_has_been_moved_rate"] = summary_dict["mug_has_been_moved"] / num_demos
        summary_dict["mug_has_been_in_drawer_rate"] = (
            summary_dict["mug_has_been_in_drawer"] / num_demos
        )
        summary_dict["mug_has_been_in_wrong_drawer_rate"] = (
            summary_dict["mug_has_been_in_wrong_drawer"] / num_demos
        )

        # Update the summary in the eval dict
        print(f"Summary of closed loop evaluation:")
        print(
            f"{summary_dict['success']} of {num_demos} demos succeeded,"
            f" success rate: {summary_dict['success_rate']}"
        )
        print(summary_dict)
        self.eval_dict["summary"] = summary_dict

        self.maybe_write_eval_file()

        return self.eval_dict

    def log_to_wandb(self) -> None:
        """
        Logs the evaluation summary to wandb.
        NOTE: For this to work, the wandb run must be initialized previously.
        """
        summary_dict = self.eval_dict["summary"]

        # Get the step id for wandb
        number_of_demos = sum(summary_dict["demo"].values())
        if not self.wandb_step_id:
            step_id = number_of_demos
        else:
            step_id = self.wandb_step_id

        # Log to wandb
        wandb.log({"closed_loop/num_demos": number_of_demos}, step=step_id)
        wandb.log(
            {"closed_loop/mug_has_been_lifted_rate": summary_dict["mug_has_been_lifted_rate"]},
            step=step_id,
        )
        wandb.log(
            {"closed_loop/mug_has_been_moved_rate": summary_dict["mug_has_been_moved_rate"]},
            step=step_id,
        )
        wandb.log(
            {
                "closed_loop/mug_has_been_in_drawer_rate": summary_dict[
                    "mug_has_been_in_drawer_rate"
                ]
            },
            step=step_id,
        )
        wandb.log(
            {
                "closed_loop/mug_has_been_in_wrong_drawer_rate": summary_dict[
                    "mug_has_been_in_wrong_drawer_rate"
                ]
            },
            step=step_id,
        )
        wandb.log({"closed_loop/success_rate": summary_dict["success_rate"]}, step=step_id)

    def _get_mug_position(self, env: gym.Env) -> NDArray[float]:
        """
        Get the positions of the mug in the environment.

        Args:
            env (gym.Env): The environment.

        Returns:
            NDArray[float]: An 1x3 array of mug position.
        """
        curr_state = env.unwrapped.scene.get_state(is_relative=True)
        return curr_state["rigid_object"]["target_mug"]["root_pose"][0, :3].cpu().numpy()

    def _get_drawer_position(self, env: gym.Env, drawer_name: str) -> NDArray[float]:
        curr_state = env.unwrapped.scene.get_state(is_relative=True)
        return curr_state["rigid_object"][drawer_name]["root_pose"][0, :3].cpu().numpy()

    def _mug_is_in_drawer(
        self, mug_position: NDArray[float], drawer_position: NDArray[float]
    ) -> bool:
        """
        Determines if the demo is successful based on the evaluation results.

        Returns:
            bool: True if the demo is successful, False otherwise.
        """
        # Define bounds relative to the drawer position
        # NOTE: drawer z position is defined at the bottom of the drawer
        bounds_x_lower = drawer_position[0] - self.drawer_size[0] / 2
        bounds_x_upper = drawer_position[0] + self.drawer_size[0] / 2
        bounds_y_lower = drawer_position[1] - self.drawer_size[1] / 2
        bounds_y_upper = drawer_position[1] + self.drawer_size[1] / 2
        bounds_z_lower = drawer_position[2] - 1e-2  # 1 cm tolerance
        bounds_z_upper = drawer_position[2] + self.drawer_size[2]

        # Check if object is within bounds
        in_x_bounds = bounds_x_lower < mug_position[0] < bounds_x_upper
        in_y_bounds = bounds_y_lower < mug_position[1] < bounds_y_upper
        in_z_bounds = bounds_z_lower < mug_position[2] < bounds_z_upper

        return bool(in_x_bounds and in_y_bounds and in_z_bounds)

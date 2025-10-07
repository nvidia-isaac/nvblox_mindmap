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

JSON_INDENT = 4


class CubeStackingEvaluator(ArmEvaluatorBase):
    """
    The purpose of this class is to evaluate the performance of the cube stacking task
    by tracking the number of cubes lifted and moved,
    as well as the number of cubes stacked.
    """

    def __init__(
        self,
        task: Tasks,
        eval_file_path: os.path = None,
        wandb_step_id: int = None,
        num_cubes: int = 3,
        cube_side_length: float = 0.045,
        gripper_closedness_threshold: float = 0.5,
        checkpoint_name=None,
    ) -> None:
        """
        Initializes the CubeStackingEvaluator object.

        Args:
            eval_file_path (os.path, optional): The path where the the evaluation file should be stored.
                                                Defaults to None (which means no evaluation file will be stored).
            wandb_step_id (int, optional): The ID of the current step in the W&B run. Defaults to None.
            num_cubes (int, optional): The number of cubes to stack. Defaults to 3.
            cube_side_length (float, optional): The side length of the cubes. Defaults to 0.045.
            gripper_closedness_threshold (float, optional): The threshold for the closeness of the gripper. Defaults to 0.5.
            checkpoint_name (str, optional): Name of checkpoint used for evaluation.
        """
        assert task == Tasks.CUBE_STACKING
        super().__init__(task, eval_file_path, wandb_step_id, checkpoint_name)
        self.num_cubes = num_cubes
        self.cube_side_length = cube_side_length
        self.gripper_closedness_threshold = gripper_closedness_threshold

        ## Setting up thresholds for evaluation the cube stacking task:
        # Cubes are considered to be moved if they are moved more than half a cube side length in xy.
        self.min_distance_xy_moved_thresh = self.cube_side_length / 2.0
        # Cubes are considered to be lifted if they are lifted more than half a cube side length in z.
        self.min_distance_z_lifted_thresh = self.cube_side_length / 2.0
        # Cubes should be separated at least by a cube side length in height to be considered stacked.
        self.min_distance_z_stacked_thresh = (
            self.cube_side_length * 0.8
        )  # subtracting 20 % to be conservative

    def init_demo(self, demo_name: str, retry_idx: int, env: gym.Env) -> None:
        """
        Initializes the demo for the cube stacking task. Should be called at the beginning of each new demo.

        Args:
            demo_name (str): The name of the demo.
            retry_idx (int): The index of the current retry.
            env (gym.Env): The environment in which the demo is being initialized.
        """
        self.demo_name = demo_name
        self.retry_idx = retry_idx
        # Get the initial positions of the cubes.
        self.initial_cube_positions = self._get_cube_positions(env)
        # Reset the evaluation variables.
        self.cubes_have_been_lifted = np.array([False] * self.num_cubes)
        self.cubes_have_been_moved = np.array([False] * self.num_cubes)
        self.max_num_stacked_cubes = 0
        self.max_num_stacked_cubes_with_open_gripper = 0
        self.current_num_stacked_cubes = 0

    def evaluate_step(self, observed_state: ArmEmbodimentRobotState, env: gym.Env) -> None:
        """
        Evaluates the current state of the cube stacking task.

        Args:
            observed_state (State): The observed state of the environment.
            env (gym.Env): The environment in which the cube stacking task is being evaluated.

        Returns:
            bool: A tuple containing:
                - A 1D boolean array of size `num_cubes` indicating whether the cubes are currently lifted.
                - A 1D boolean array of size `num_cubes` indicating whether the cubes are currently moved.
                - An integer indicating the number of cubes currently stacked.
        """
        # Get the initial cube positions
        initial_cube_z = self.initial_cube_positions[:, 2]
        initial_cube_xy = self.initial_cube_positions[:, :2]

        # Get the current cube positions
        cube_positions = self._get_cube_positions(env)
        cubes_z = cube_positions[:, 2]
        cubes_xy = cube_positions[:, :2]

        # Check constant number of cubes
        assert len(self.initial_cube_positions) == len(cube_positions)

        # Check whether the cubes have been lifted
        delta_cubes_z = cubes_z - initial_cube_z
        cubes_are_lifted = delta_cubes_z > self.min_distance_z_lifted_thresh
        self.cubes_have_been_lifted = np.logical_or(self.cubes_have_been_lifted, cubes_are_lifted)

        # Check whether the cubes have been moved
        delta_cubes_xy = np.linalg.norm(cubes_xy - initial_cube_xy, axis=-1)
        cubes_are_moved = delta_cubes_xy > self.min_distance_xy_moved_thresh
        self.cubes_have_been_moved = np.logical_or(self.cubes_have_been_moved, cubes_are_moved)

        # Check how many cubes are stacked on top of each other
        num_stacked_cubes = self._get_num_stacked_cubes(cube_positions)
        if num_stacked_cubes > self.max_num_stacked_cubes:
            self.max_num_stacked_cubes = num_stacked_cubes

        # Separately track the gripper was open when reach the maximum number of stacked cubes
        if (
            self._gripper_is_open(observed_state)
            and num_stacked_cubes > self.max_num_stacked_cubes_with_open_gripper
        ):
            self.max_num_stacked_cubes_with_open_gripper = num_stacked_cubes

        # Save the *current* number of stacked cubes
        self.current_num_stacked_cubes = num_stacked_cubes

        return cubes_are_lifted, cubes_are_moved, num_stacked_cubes

    def finalize_demo(self, observed_state: ArmEmbodimentRobotState, env) -> None:
        """
        Finalizes the evaluation of a demo for the cube stacking task.
        Saves the most recent evaluation results to a file.

        Args:
            observed_state (State): The observed state of the environment.
            env (gym.Env): The environment in which the cube stacking task is being evaluated.
        """
        self.evaluate_step(observed_state, env)
        demo_key = f"{self.demo_name}_{self.retry_idx}"
        success = self.max_num_stacked_cubes_with_open_gripper == self.num_cubes
        self.eval_dict[demo_key] = {
            "demo": self.demo_name,
            "success": success,
            "num_stacked_cubes": int(self.current_num_stacked_cubes),
            "cubes_have_been_lifted": int(np.sum(self.cubes_have_been_lifted)),
            "cubes_have_been_moved": int(np.sum(self.cubes_have_been_moved)),
            "max_num_stacked_cubes": int(self.max_num_stacked_cubes),
            "max_num_stacked_cubes_with_open_gripper": int(
                self.max_num_stacked_cubes_with_open_gripper
            ),
        }

        print(f"Closed loop success of {self.demo_name}: {success}")

        self.maybe_write_eval_file()

    def summarize_demos(self) -> Dict:
        """
        Summarizes the evaluation of the cube stacking task for multiple demos.
        Saves the evaluation of all demos together with the summary to a file and returns eval dict.
        """
        init_counts_dict = {i: 0 for i in range(0, self.num_cubes + 1)}
        summary_dict = {
            "demo": {},
            "success": {True: 0, False: 0},
            "cubes_are_lifted": init_counts_dict.copy(),
            "cubes_are_moved": init_counts_dict.copy(),
            "num_stacked_cubes": init_counts_dict.copy(),
            "cubes_have_been_lifted": init_counts_dict.copy(),
            "cubes_have_been_moved": init_counts_dict.copy(),
            "max_num_stacked_cubes": init_counts_dict.copy(),
            "max_num_stacked_cubes_with_open_gripper": init_counts_dict.copy(),
        }

        for key, demo_eval_dict in self.eval_dict.items():
            if key in ["summary", "metadata"]:
                continue
            for key, value in demo_eval_dict.items():
                self._count_occurrences(summary_dict[key], value)

        # Compute mean values
        num_demos = sum(summary_dict["demo"].values())
        summary_dict["num_demos"] = num_demos
        summary_dict["mean_num_lifted_cubes"] = self._mean_count_per_demo(
            summary_dict["cubes_have_been_lifted"]
        )
        summary_dict["mean_num_moved_cubes"] = self._mean_count_per_demo(
            summary_dict["cubes_have_been_moved"]
        )
        summary_dict["mean_num_stacked_cubes"] = self._mean_count_per_demo(
            summary_dict["max_num_stacked_cubes"]
        )
        summary_dict["mean_num_stacked_cubes_with_open_gripper"] = self._mean_count_per_demo(
            summary_dict["max_num_stacked_cubes_with_open_gripper"]
        )
        summary_dict["full_stack_at_demo_end_rate"] = (
            summary_dict["num_stacked_cubes"][self.num_cubes] / num_demos
        )
        summary_dict["success_rate"] = self._mean_count_per_demo(summary_dict["success"])

        # Update the summary in the eval dict
        print(f"Summary of closed loop evaluation:")
        print(
            f"{summary_dict['success'][True]} of {num_demos} demos succeeded,"
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
            {"closed_loop/mean_num_lifted_cubes": summary_dict["mean_num_lifted_cubes"]},
            step=step_id,
        )
        wandb.log(
            {"closed_loop/mean_num_moved_cubes": summary_dict["mean_num_moved_cubes"]}, step=step_id
        )
        wandb.log(
            {"closed_loop/mean_num_stacked_cubes": summary_dict["mean_num_stacked_cubes"]},
            step=step_id,
        )
        wandb.log(
            {
                "closed_loop/mean_num_stacked_cubes_with_open_gripper": summary_dict[
                    "mean_num_stacked_cubes_with_open_gripper"
                ]
            },
            step=step_id,
        )
        wandb.log({"closed_loop/success_rate": summary_dict["success_rate"]}, step=step_id)
        wandb.log(
            {
                "closed_loop/full_stack_at_demo_end_rate": summary_dict[
                    "full_stack_at_demo_end_rate"
                ]
            },
            step=step_id,
        )

    def _get_num_stacked_cubes(self, cube_positions: NDArray[float]) -> int:
        """
        Calculates the number of cubes on the highest stack.

        Args:
            cube_positions (NDArray[float]): A Nx3 array of cube positions.

        Returns:
            int: The number of cubes on the highest stack (including the cube on the bottom).
        """
        num_cubes_on_highest_stack = 0
        cubes_z = cube_positions[:, 2]
        for i in range(self.num_cubes):
            num_cubes_on_stack = 1  # also count the cube on the bottom
            for j in range(i + 1, self.num_cubes):  # test every pair of cubes
                distance_z = abs(cubes_z[i] - cubes_z[j])
                if self._cubes_are_stacked(distance_z):
                    num_cubes_on_stack += 1
            if num_cubes_on_stack > num_cubes_on_highest_stack:
                num_cubes_on_highest_stack = num_cubes_on_stack
        return num_cubes_on_highest_stack

    def _cubes_are_stacked(self, distance_z: float) -> bool:
        """
        Determines if two cubes are stacked based on distance in the z axis.

        Args:
            distance_z (float): The distance between the centers of the cubes in the z axis.

        Returns:
            bool: True if the cubes are stacked, False otherwise.
        """
        return distance_z > self.min_distance_z_stacked_thresh

    def _get_cube_positions(self, env: gym.Env) -> NDArray[float]:
        """
        Get the positions of the cubes in the environment.

        Args:
            env (gym.Env): The environment.

        Returns:
            NDArray[float]: An Nx3 array of cube positions.
        """
        curr_state = env.unwrapped.scene.get_state(is_relative=True)
        return np.array(
            [
                curr_state["rigid_object"][f"cube_{idx + 1}"]["root_pose"][0, :3].cpu().numpy()
                for idx in range(self.num_cubes)
            ]
        )

    def _mean_count_per_demo(self, count_dict: Dict[int, int]) -> float:
        """
        Calculates the average count per demo.

        Args:
            count_dict (Dict[int, int]): A dictionary containing the number of demos for which the count occurred.

        Returns:
            float: The average count per demo.
        """
        number_of_demos = float(sum(count_dict.values()))
        total_count = sum(
            [count_in_demo * demo_count for count_in_demo, demo_count in count_dict.items()]
        )
        average_count_per_demo = total_count / number_of_demos
        return average_count_per_demo

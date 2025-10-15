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
from datetime import datetime
import json
import os
from typing import Dict, Hashable

import gymnasium as gym
import wandb

from mindmap.embodiments.state_base import RobotStateBase
from mindmap.tasks.task_success import get_task_outcome
from mindmap.tasks.tasks import Tasks

JSON_INDENT = 4


class EvaluatorBase(ABC):
    """
    Base class for all evaluators. An evaluator tracks the performance of a task over a series of demos.
    """

    def __init__(
        self,
        task: Tasks,
        eval_file_path: os.path = None,
        wandb_step_id: int = None,
        checkpoint_name=None,
    ) -> None:
        """
        Initializes the EvaluatorBase object.

        Args:
            eval_file_path (os.path, optional): The path where the the evaluation file should be stored.
                                                Defaults to None (which means no evaluation file will be stored).
            wandb_step_id (int, optional): The ID of the current step in the W&B run. Defaults to None.
            checkpoint_name (str, optional): Name of checkpoint used for evaluation.
        """
        self.task = task
        if eval_file_path is not None:
            assert os.path.exists(os.path.dirname(eval_file_path))
        self.wandb_step_id = wandb_step_id
        self.eval_file_path = eval_file_path
        self.eval_dict = {}
        self.eval_dict["metadata"] = {
            "checkpoint_name": checkpoint_name,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    @abstractmethod
    def init_demo(self, demo_name: str, retry_idx: int, env: gym.Env) -> None:
        """
        Initializes the demo for the cube stacking task. Should be called at the beginning of each new demo.

        Args:
            demo_name (str): The name of the demo.
            retry_idx (int): The index of the current retry.
            env (gym.Env): The environment in which the demo is being initialized.
        """
        pass

    @abstractmethod
    def evaluate_step(self, observed_state: RobotStateBase, env: gym.Env) -> None:
        """
        Evaluates the current state of the cube stacking task.

        Args:
            observed_state (State): The observed state of the environment.
            env (gym.Env): The environment in which the cube stacking task is being evaluated.
        """
        pass

    @abstractmethod
    def finalize_demo(self, observed_state: RobotStateBase, env) -> None:
        """
        Finalizes the evaluation of a demo for the cube stacking task.
        Saves the most recent evaluation results to a file.

        Args:
            observed_state (State): The observed state of the environment.
            env (gym.Env): The environment in which the cube stacking task is being evaluated.
        """
        pass

    @abstractmethod
    def summarize_demos(self) -> Dict:
        pass

    @abstractmethod
    def log_to_wandb(self) -> None:
        pass

    def _count_occurrences(self, counts_dict: dict, value: Hashable) -> None:
        """
        Count the occurrences of a given value in a dictionary.

        Args:
            counts_dict (dict): A dictionary to keep track of the count of each value.
            value (Hashable): The value to count the occurrences of.
        """
        if value in counts_dict:
            counts_dict[value] += 1
        else:
            counts_dict[value] = 1

    def maybe_write_eval_file(self):
        """
        If the evaluation file is set, the eval dict will be written to it.
        """
        if self.eval_file_path is not None:
            with open(self.eval_file_path, "w") as json_file:
                json.dump(self.eval_dict, json_file, indent=JSON_INDENT)


class BasicEvaluator(EvaluatorBase):
    """
    A basic evaluator that just checks the task success (given by get_task_outcome).
    """

    def init_demo(self, demo_name: str, retry_idx: int, env: gym.Env) -> None:
        self.demo_name = demo_name
        self.retry_idx = retry_idx

    def evaluate_step(self, observed_state: RobotStateBase, env: gym.Env) -> None:
        pass

    def finalize_demo(self, observed_state: RobotStateBase, env) -> None:
        success = get_task_outcome(self.task, env.unwrapped).item()
        demo_key = f"{self.demo_name}_{self.retry_idx}"
        self.eval_dict[demo_key] = {
            "demo": self.demo_name,
            "success": success,
        }
        print(f"Closed loop success of {self.demo_name}: {success}")
        self.maybe_write_eval_file()

    def summarize_demos(self) -> Dict:
        """
        Summarizes the evaluation of the task for multiple demos.
        Saves the evaluation of all demos together with the summary to a file and returns eval dict.
        """
        summary_dict = {
            "demos": [],
            "success": 0,
        }

        for key, demo_eval_dict in self.eval_dict.items():
            if key in ["summary", "metadata"]:
                continue
            summary_dict["demos"].append(demo_eval_dict["demo"])
            if demo_eval_dict["success"]:
                summary_dict["success"] += 1

        # Compute mean values
        num_demos = len(summary_dict["demos"])
        summary_dict["num_demos"] = num_demos
        summary_dict["success_rate"] = summary_dict["success"] / num_demos

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
        number_of_demos = len(summary_dict["demos"])
        if not self.wandb_step_id:
            step_id = number_of_demos
        else:
            step_id = self.wandb_step_id

        # Log to wandb
        wandb.log({"closed_loop/num_demos": number_of_demos}, step=step_id)
        wandb.log({"closed_loop/success_rate": summary_dict["success_rate"]}, step=step_id)

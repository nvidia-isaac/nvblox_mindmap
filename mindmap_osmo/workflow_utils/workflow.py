# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from abc import ABC
import argparse
import subprocess
from typing import Any, Dict, List, Type

from tap import Tap
import yaml

from mindmap_osmo.tasks.base_task import BaseTask
from mindmap_osmo.workflow_utils.workflow_constants import (
    EXEC_TIMEOUT,
    PLATFORM_TO_POOL,
    QUEUE_TIMEOUT,
    WORKFLOW_FILE_PATH,
)
from mindmap_osmo.workflow_utils.workflow_types import OsmoWorkflowType


# Adding a representer to convert the file contents to a block literal string.
class block_literal_str(str):
    pass


def block_literal_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(block_literal_str, block_literal_representer)


class Workflow(ABC):
    def __init__(
        self,
        osmo_workflow_type: OsmoWorkflowType,
        workflow_args: argparse.Namespace,
        task_cls_list: List[Type[BaseTask]],
        app_args_list: List[Tap],
    ) -> None:
        """Class defining the workflow generation and submission.
        Can hold multiple tasks to run in the workflow.

        Args:
            osmo_workflow_type (OsmoWorkflowType): Type of workflow
            workflow_args (argparse.Namespace): Arguments specific to workflow configuration
            task_cls_list (List[Type[BaseTask]]): List of task classes to include in workflow
            app_args_list (List[Tap]): List of application arguments for each task
        """
        self.osmo_workflow_type = osmo_workflow_type
        self.workflow_args = workflow_args
        self.task_cls_list = task_cls_list
        self.app_args_list = app_args_list
        assert len(self.app_args_list) == len(self.task_cls_list)
        assert len(self.app_args_list) > 0

    def generate_workflow(self) -> dict:
        """Creates workflow dictionary and writes it to disk as YAML."""
        workflow_dict = self._create_workflow_dict()

        with open(WORKFLOW_FILE_PATH, "w") as f:
            yaml.dump(workflow_dict, f, default_flow_style=False, sort_keys=False, default_style="")

        # Print the file contents to console for quick verification
        with open(WORKFLOW_FILE_PATH, "r") as f:
            print(f.read())
        print(f"Workflow YAML generated at {WORKFLOW_FILE_PATH}")

        return workflow_dict

    def submit_workflow(self, dry_run: bool) -> bool:
        """Submits the workflow to the OSMO platform."""
        pool = PLATFORM_TO_POOL[self.workflow_args.platform]
        if dry_run:
            if not self._run_subprocess(
                ["osmo", "workflow", "validate", "--pool", pool, WORKFLOW_FILE_PATH]
            ):
                print(f"Workflow validation failed.")
                return False
        else:
            if not self._run_subprocess(
                ["osmo", "workflow", "submit", "--pool", pool, WORKFLOW_FILE_PATH]
            ):
                print(f"Workflow submission failed.")
                return False
        return True

    def _create_workflow_dict(self) -> Dict[str, Any]:
        """Creates and returns the workflow dictionary using abstract methods defined by child workflow classes."""
        workflow_dict = {
            "workflow": {
                "name": self._get_workflow_name(),
                "resources": {},
                "timeout": {"queue_timeout": QUEUE_TIMEOUT, "exec_timeout": EXEC_TIMEOUT},
                "tasks": [],
            }
        }
        tasks = self._get_tasks()
        for task in tasks:
            workflow_dict["workflow"]["tasks"].append(task.create_task_dict())
            workflow_dict["workflow"]["resources"][
                task.get_task_name()
            ] = task.create_resource_dict()

        return workflow_dict

    def _get_tasks(self) -> List[BaseTask]:
        """Returns a list of initialized task objects for the workflow."""
        tasks = []
        for task_cls, app_args in zip(self.task_cls_list, self.app_args_list):
            assert issubclass(task_cls, BaseTask)
            tasks.append(task_cls(self.osmo_workflow_type, self.workflow_args, app_args))
        return tasks

    def _get_workflow_name(self) -> str:
        """Returns the workflow name."""
        # Name wandb and OSMO workflow the same
        workflow_name = self.app_args_list[0].wandb_name
        for app_args in self.app_args_list:
            # Make sure all wandb runs of all
            # osmo tasks have the same name as the workflow.
            assert app_args.wandb_name == workflow_name
        return workflow_name

    @staticmethod
    def _run_subprocess(command: List[str], check: bool = True) -> bool:
        """Runs a subprocess command and returns True if it succeeds, False otherwise."""
        result = subprocess.run(command, check=check)
        return result.returncode == 0

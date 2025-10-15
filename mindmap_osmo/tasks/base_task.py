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
import argparse
import os
from typing import Any, Dict, List

from tap import Tap
import yaml

from mindmap_osmo.workflow_utils.inferred_args import get_inferred_args
from mindmap_osmo.workflow_utils.workflow_constants import (
    CHECKPOINT_DATASET_NAME,
    HDF5_DATASET_NAME,
    MAX_STORAGE_ON_PLATFORM_GB,
    SWIFT_URL,
)
from mindmap_osmo.workflow_utils.workflow_types import DatasetType, OsmoTaskType, OsmoWorkflowType


# Adding a representer to convert the file contents to a block literal string.
class block_literal_str(str):
    pass


def block_literal_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(block_literal_str, block_literal_representer)


class BaseTask(ABC):
    """Base class for all task types.
    Provides common functionality and defines abstract methods that
    must be implemented by child task classes."""

    def __init__(
        self,
        osmo_workflow_type: OsmoWorkflowType,
        osmo_task_type: OsmoTaskType,
        workflow_args: argparse.Namespace,
        app_args: Tap,
    ) -> None:
        """Base task class constructor.

        Args:
            osmo_workflow_type (OsmoWorkflowType): Type of workflow that this task belongs to.
            osmo_task_type (OsmoTaskType): Type of this osmo task.
            workflow_args (argparse.Namespace): Arguments specific to workflow configuration
            app_args (Tap): Arguments specific to application configuration
        """
        self.osmo_workflow_type = osmo_workflow_type
        self.osmo_task_type = osmo_task_type
        self.workflow_args = workflow_args
        self.app_args = app_args
        self.inferred_args = get_inferred_args(
            osmo_workflow_type=self.osmo_workflow_type,
            workflow_args=self.workflow_args,
            osmo_task_type=self.osmo_task_type,
            app_args=self.app_args,
            dataset_indices=self._get_dataset_indices(),
        )

        max_storage = MAX_STORAGE_ON_PLATFORM_GB[self.workflow_args.platform]
        assert max_storage >= self.inferred_args.storage_gb, (
            f"Not enough storage on platform {self.workflow_args.platform}: "
            f"{max_storage}Gi < {self.inferred_args.storage_gb}Gi"
        )

    def create_task_dict(self) -> Dict[str, Any]:
        """Creates and returns the task dictionary using abstract methods defined by child workflow classes."""
        task_dict = {
            "name": self.get_task_name(),
            "resource": self.get_task_name(),
            "image": f"nvcr.io/PLACEHOLDER_NGC_PATH:{self.workflow_args.image_tag}",
            "environment": self._get_environment(),
            "credentials": self._get_credentials(),
            "downloadType": "download",
            "inputs": self._get_inputs(),
            "command": ["bash"],
            "args": ["/tmp/entry.sh"],
            "files": [
                {"path": "/tmp/entry.sh", "contents": block_literal_str(self._get_run_script())}
            ],
        }

        if self._get_output() is not None:
            task_dict["outputs"] = self._get_output()

        if self._get_periodic_checkpoint() is not None:
            task_dict["checkpoint"] = self._get_periodic_checkpoint()

        return task_dict

    def create_resource_dict(self) -> Dict[str, Any]:
        return {
            "cpu": self.inferred_args.num_cpus,
            "gpu": self.inferred_args.num_gpus,
            "memory": f"{self.inferred_args.memory_gb}Gi",
            "storage": f"{self.inferred_args.storage_gb}Gi",
            "platform": self.workflow_args.platform.value,
        }

    @staticmethod
    def _get_environment() -> Dict[str, str]:
        """Returns environment dictionary for the osmo task."""
        environment_dict = {
            "ACCEPT_EULA": "Y",
            "NO_NUCLEUS": "Y",
            "OMNI_SERVER": "isaac-dev.ov.nvidia.com",
        }
        return environment_dict

    @staticmethod
    def _get_credentials() -> Dict[str, Dict[str, str]]:
        """Returns credentials dictionary for the osmo task."""
        credentials_dict = {
            "omni_svc": {"OMNI_USER": "omni_user", "OMNI_PASS": "omni_pass"},
            "wandb-auth": {"WANDB_API_KEY": "wandb_pass"},
        }
        return credentials_dict

    def _get_inputs(self) -> List[Dict[str, Any]]:
        """Returns input list for the osmo task (defining the input datasets)."""
        input = []
        for dataset_type in self._get_dataset_types():
            input.append(self._get_dataset_dict(dataset_type))
        return input

    def _get_dataset_dict(self, dataset_type: DatasetType) -> Dict[str, Any]:
        """Returns the dataset dictionary for the given dataset type."""
        if dataset_type == DatasetType.CHECKPOINT:
            return self._get_checkpoint_dataset()
        elif dataset_type == DatasetType.CHECKPOINT_SWIFT:
            return self._get_checkpoint_swift_dataset(self.app_args.checkpoint)
        elif dataset_type == DatasetType.HDF5:
            return self._get_hdf5_dataset()
        elif dataset_type == DatasetType.DATA:
            return self._get_input_dataset()
        elif dataset_type == DatasetType.TASK:
            return self._get_task_dataset()
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

    def _get_dataset_indices(self) -> Dict[str, int]:
        """Returns the dataset indices for the given dataset types."""
        dataset_indices = {}
        for idx, dataset_type in enumerate(self._get_dataset_types()):
            dataset_indices[dataset_type.name] = idx
        for dataset_type in DatasetType:
            if dataset_type.name not in dataset_indices:
                dataset_indices[dataset_type.name] = None
        return dataset_indices

    @abstractmethod
    def _get_dataset_types(self) -> List[DatasetType]:
        """Returns the dataset types for the osmo task."""
        pass

    @abstractmethod
    def _get_output(self) -> List[Dict[str, Any]]:
        """Returns output dictionary for the osmo task (defining the output datasets)."""
        pass

    def _get_periodic_checkpoint(self) -> List[Dict[str, Any]]:
        """Override to enable saving of periodic checkpoints."""
        return None

    @abstractmethod
    def _get_run_script(self) -> str:
        """Returns the contents of the entry script for the osmo task."""
        pass

    def _get_checkpoint_dataset(self) -> Dict[str, str]:
        """Returns the checkpoint dataset dictionary for the osmo task."""
        regex = ".*args.json"
        if self.app_args.checkpoint is not None:
            regex += f"|{self.app_args.checkpoint}"
        if self.inferred_args.fpn_file_name is not None:
            regex += f"|{self.inferred_args.fpn_file_name}"
        return {"dataset": {"name": CHECKPOINT_DATASET_NAME, "regex": regex}}

    def _get_checkpoint_swift_dataset(self, checkpoint_path: str) -> Dict[str, str]:
        """Returns the checkpoint dataset dictionary for the osmo task."""
        checkpoint_dir = os.path.split(checkpoint_path)[:-1]
        return {"url": f"{SWIFT_URL}/{CHECKPOINT_DATASET_NAME}/{'/'.join(checkpoint_dir)}"}

    @staticmethod
    @abstractmethod
    def get_task_name() -> str:
        """Returns the task name for the osmo task."""
        pass

    @staticmethod
    @abstractmethod
    def get_input_task_name() -> str:
        """Returns the name of the input task for the workflow."""
        pass

    def _get_hdf5_dataset(self) -> Dict[str, str]:
        """Returns the HDF5 dataset dictionary for the osmo task."""
        return {"dataset": {"name": HDF5_DATASET_NAME, "regex": self.inferred_args.hdf5_file_name}}

    def _get_task_dataset(self) -> Dict[str, str]:
        """Returns the input task as a dataset dictionary for the osmo task."""
        assert self.get_input_task_name() is not None
        return {"task": self.get_input_task_name()}

    def _get_input_dataset(self) -> Dict[str, str]:
        """Returns the input dataset dictionary for the osmo task."""
        return {
            "dataset": {
                "name": self.inferred_args.input_dataset_name,
                "regex": self.inferred_args.input_dataset_regex,
            }
        }

    def _get_output_dataset(self) -> Dict[str, str]:
        """Returns the output dataset dictionary for the osmo task."""
        return {
            "dataset": {
                "name": self.inferred_args.output_dataset_name,
                "metadata": ["metadata.yaml"],
            }
        }

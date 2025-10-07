# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import argparse
from typing import Any, Dict, List

from mindmap.cli.args import ClosedLoopAppArgs, DataGenAppArgs
from mindmap.data_loading.data_types import DataType
from mindmap_osmo.tasks.base_task import BaseTask
from mindmap_osmo.workflow_utils.arg_parsing import (
    get_log_to_file_arg_string,
    get_non_default_args_str,
)
from mindmap_osmo.workflow_utils.code_snippets import init_script_snippet, tar_demos_snippet
from mindmap_osmo.workflow_utils.inferred_args import InferredArgs, fpn_required
from mindmap_osmo.workflow_utils.workflow_types import DatasetType, OsmoTaskType, OsmoWorkflowType


def run_isaaclab_datagen_snippet(app_args: DataGenAppArgs, inferred_args: InferredArgs) -> str:
    """Get the command string for running the IsaacLab datagen."""
    command_str = f"""
python run_isaaclab_datagen.py \\
    --num_envs 1 \\
    --hdf5_file {inferred_args.hdf5_osmo_path} \\
    --output_dir {{{{output}}}}"""

    # Update the local FPN checkpoint path with the OSMO path.
    if fpn_required(OsmoTaskType.DATAGEN, app_args):
        app_args.fpn_checkpoint = inferred_args.fpn_osmo_path

    command_str += get_non_default_args_str(app_args, DataGenAppArgs)
    command_str += get_log_to_file_arg_string("datagen.log")
    return command_str


def run_isaaclab_dataset_validation_snippet(
    app_args: ClosedLoopAppArgs, inferred_args: InferredArgs
) -> str:
    command_str = f"""
python run_validate_demos.py \\
    --num_envs 1 \\
    --hdf5_file {inferred_args.hdf5_osmo_path} \\
    --task {app_args.task.value} \\
    --dataset {{{{output}}}}"""

    command_str += get_non_default_args_str(app_args, DataGenAppArgs)

    # Validation runs under closed loop mode, but with the same demos as the data generation task.
    # We therefore need to change the name of this argument.
    command_str = command_str.replace("--demos_datagen", "--demos_closed_loop")
    return command_str


def add_datagen_metadata_snippet(demos: str, osmo_dataset_note: str) -> str:
    """Get the command string for adding datagen metadata."""
    return f"""
# Add some metadata for the osmo dataset
echo "demos: {demos}" >> {{{{output}}}}/metadata.yaml
echo "osmo_dataset_note: {osmo_dataset_note}" >> {{{{output}}}}/metadata.yaml
"""


class DatagenTask(BaseTask):
    def __init__(
        self,
        osmo_workflow_type: OsmoWorkflowType,
        workflow_args: argparse.Namespace,
        app_args: argparse.Namespace,
    ) -> None:
        super().__init__(
            osmo_workflow_type=osmo_workflow_type,
            osmo_task_type=OsmoTaskType.DATAGEN,
            workflow_args=workflow_args,
            app_args=app_args,
        )

    @staticmethod
    def get_task_name() -> str:
        """Returns the task name for the workflow."""
        return OsmoTaskType.DATAGEN.value

    @staticmethod
    def get_input_task_name() -> str:
        """Returns the name of the input task for the workflow."""
        return None

    def _get_dataset_types(self) -> List[DatasetType]:
        dataset_types = []
        dataset_types.append(DatasetType.HDF5)
        if fpn_required(OsmoTaskType.DATAGEN, self.app_args):
            dataset_types.append(DatasetType.CHECKPOINT)
        return dataset_types

    def _get_output(self) -> List[Dict[str, Any]]:
        if (
            self.osmo_workflow_type == OsmoWorkflowType.E2E
            and not self.workflow_args.upload_e2e_dataset
        ):
            # Per default, we do not upload the E2E dataset to an OSMO dataset.
            return None
        else:
            output = [
                self._get_output_dataset(),
            ]
            return output

    def _get_run_script(self) -> str:
        file_contents = init_script_snippet()
        file_contents += run_isaaclab_datagen_snippet(self.app_args, self.inferred_args)
        if self.app_args.validate_demos_with_gt_poses:
            file_contents += run_isaaclab_dataset_validation_snippet(
                self.app_args, self.inferred_args
            )
        file_contents += tar_demos_snippet(self.inferred_args.num_cpus)
        file_contents += add_datagen_metadata_snippet(
            self.app_args.demos_datagen, self.workflow_args.osmo_dataset_note
        )
        return file_contents

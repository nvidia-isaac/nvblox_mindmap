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

from tap import Tap

from mindmap.cli.args import TrainingAppArgs
from mindmap.common_utils.system import get_random_port_in_unassigned_range
from mindmap_osmo.tasks.base_task import BaseTask
from mindmap_osmo.tasks.datagen_task import DatagenTask
from mindmap_osmo.workflow_utils.arg_parsing import (
    get_log_to_file_arg_string,
    get_non_default_args_str,
)
from mindmap_osmo.workflow_utils.code_snippets import init_script_snippet, untar_demos_snippet
from mindmap_osmo.workflow_utils.inferred_args import InferredArgs, fpn_required
from mindmap_osmo.workflow_utils.workflow_constants import (
    CHECKPOINT_DATASET_NAME,
    NNODES,
    PERIODIC_CHECKPOINT_FREQUENCY,
    SWIFT_URL,
)
from mindmap_osmo.workflow_utils.workflow_types import DatasetType, OsmoTaskType, OsmoWorkflowType


def run_training_snippet(app_args: TrainingAppArgs, inferred_args: InferredArgs) -> str:
    """Get the command string for running the training."""
    port = get_random_port_in_unassigned_range()
    command_str = f"""
torchrun --standalone --nnodes {NNODES} --nproc_per_node  {inferred_args.num_gpus} --max-restarts 3 \\
    --master_port {port} run_training.py \\
    --dataset {inferred_args.input_dataset_path}"""

    # Update the local checkpoint paths with the OSMO path.
    app_args.checkpoint = inferred_args.checkpoint_osmo_path
    if fpn_required(OsmoTaskType.TRAINING, app_args):
        app_args.fpn_checkpoint = inferred_args.fpn_osmo_path

    command_str += get_non_default_args_str(app_args, TrainingAppArgs)
    command_str += get_log_to_file_arg_string("training.log")
    return command_str


class TrainingTask(BaseTask):
    def __init__(
        self, osmo_workflow_type: OsmoWorkflowType, workflow_args: argparse.Namespace, app_args: Tap
    ) -> None:
        super().__init__(
            osmo_workflow_type=osmo_workflow_type,
            osmo_task_type=OsmoTaskType.TRAINING,
            workflow_args=workflow_args,
            app_args=app_args,
        )

    @staticmethod
    def get_task_name() -> str:
        """Returns the task name for the workflow."""
        return OsmoTaskType.TRAINING.value

    @staticmethod
    def get_input_task_name() -> str:
        """Returns the name of the input task for the workflow."""
        return DatagenTask.get_task_name()

    def _get_dataset_types(self) -> List[DatasetType]:
        dataset_types = []
        dataset_types.append(DatasetType.HDF5)
        if self.osmo_workflow_type == OsmoWorkflowType.E2E:
            dataset_types.append(DatasetType.TASK)
        else:
            dataset_types.append(DatasetType.DATA)
        if (
            fpn_required(OsmoTaskType.TRAINING, self.app_args)
            or self.app_args.checkpoint is not None
        ):
            dataset_types.append(DatasetType.CHECKPOINT)
        return dataset_types

    def _get_output(self) -> List[Dict[str, Any]]:
        output_dict = [{"url": f"{SWIFT_URL}/{CHECKPOINT_DATASET_NAME}/{{{{workflow_id}}}}"}]
        return output_dict

    def _get_periodic_checkpoint(self) -> List[Dict[str, Any]]:
        output_dict = [
            {
                "path": "{{output}}",
                "url": f"{SWIFT_URL}/{CHECKPOINT_DATASET_NAME}/{{{{workflow_id}}}}",
                "frequency": PERIODIC_CHECKPOINT_FREQUENCY,
                "regex": ".*args.json|.*.pth",
            }
        ]
        return output_dict

    def _get_run_script(self) -> str:
        file_contents = init_script_snippet()
        file_contents += untar_demos_snippet(
            ncpus=self.inferred_args.num_cpus, demos_dir=self.inferred_args.input_dataset_path
        )
        file_contents += run_training_snippet(self.app_args, self.inferred_args)
        return file_contents

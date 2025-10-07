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

from mindmap.cli.args import ClosedLoopAppArgs, ClosedLoopMode
from mindmap.common_utils.system import get_random_port_in_unassigned_range
from mindmap_osmo.tasks.base_task import BaseTask
from mindmap_osmo.tasks.training_task import TrainingTask
from mindmap_osmo.workflow_utils.arg_parsing import (
    get_log_to_file_arg_string,
    get_non_default_args_str,
)
from mindmap_osmo.workflow_utils.code_snippets import init_script_snippet, untar_demos_snippet
from mindmap_osmo.workflow_utils.inferred_args import InferredArgs, fpn_required
from mindmap_osmo.workflow_utils.workflow_constants import SWIFT_URL
from mindmap_osmo.workflow_utils.workflow_types import DatasetType, OsmoTaskType, OsmoWorkflowType


def run_closed_loop_snippet(app_args: ClosedLoopAppArgs, inferred_args: InferredArgs) -> str:
    """Get the command string for running the closed loop policy."""
    port = get_random_port_in_unassigned_range()
    command_str = f"""
torchrun --nnodes=1 --nproc_per_node=1 --max-restarts=3 \\
    --master_port {port} run_closed_loop_policy.py \\
    --num_envs 1 \\
    --hdf5_file {inferred_args.hdf5_osmo_path} \\
    --record_camera_output_path record \\
    --record_videos"""

    if app_args.demo_mode == ClosedLoopMode.EXECUTE_GT_GOALS:
        command_str += f""" \\
    --dataset {inferred_args.input_dataset_path}"""

    # Update the local checkpoint paths with the OSMO path.
    app_args.checkpoint = inferred_args.checkpoint_osmo_path
    if fpn_required(OsmoTaskType.EVAL, app_args):
        app_args.fpn_checkpoint = inferred_args.fpn_osmo_path

    command_str += get_non_default_args_str(app_args, ClosedLoopAppArgs)
    command_str += get_log_to_file_arg_string("closed_loop.log")
    return command_str


def publish_closed_loop_eval_snippet() -> str:
    """Get the command string for publishing the closed loop eval."""
    return f"""
python3 scripts/publish_closed_loop_eval.py \\
    --eval_file_path {{{{output}}}}/closed_loop_eval.json \\
    --videos_path record \\
    --output_path {{{{output}}}}
"""


class ClosedLoopTask(BaseTask):
    """Workflow for running the closed loop policy on OSMO."""

    def __init__(
        self,
        osmo_workflow_type: OsmoWorkflowType,
        workflow_args: argparse.Namespace,
        app_args: argparse.Namespace,
    ) -> None:
        super().__init__(
            osmo_workflow_type=osmo_workflow_type,
            osmo_task_type=OsmoTaskType.EVAL,
            workflow_args=workflow_args,
            app_args=app_args,
        )

    @staticmethod
    def get_task_name() -> str:
        """Returns the task name for the workflow."""
        return OsmoTaskType.EVAL.value

    @staticmethod
    def get_input_task_name() -> str:
        """Returns the name of the input task for the workflow."""
        return TrainingTask.get_task_name()

    def _get_dataset_types(self) -> List[DatasetType]:
        dataset_types = []
        dataset_types.append(DatasetType.HDF5)
        if self.app_args.demo_mode == ClosedLoopMode.EXECUTE_GT_GOALS:
            dataset_types.append(DatasetType.DATA)
        if self.osmo_workflow_type in [OsmoWorkflowType.TRAIN_AND_EVAL, OsmoWorkflowType.E2E]:
            # Get the checkpoint from the training task.
            dataset_types.append(DatasetType.TASK)
            if fpn_required(OsmoTaskType.EVAL, self.app_args):
                dataset_types.append(DatasetType.CHECKPOINT)
        else:
            # Get checkpoint and FPN from the checkpoint dataset
            if (
                fpn_required(OsmoTaskType.EVAL, self.app_args)
                or not self.workflow_args.checkpoint_from_swiftstack
            ):
                dataset_types.append(DatasetType.CHECKPOINT)
            else:
                dataset_types.append(DatasetType.CHECKPOINT_SWIFT)

        assert not (
            fpn_required(OsmoTaskType.EVAL, self.app_args)
            and self.workflow_args.checkpoint_from_swiftstack
        ), "FPN currently not supported for swiftstack checkpoints"

        return dataset_types

    def _get_output(self) -> List[Dict[str, Any]]:
        output_dict = [{"url": f"{SWIFT_URL}/closed_loop_eval/{{{{workflow_id}}}}"}]
        return output_dict

    def _get_run_script(self) -> str:
        file_contents = init_script_snippet()
        if self.app_args.demo_mode == ClosedLoopMode.EXECUTE_GT_GOALS:
            file_contents += untar_demos_snippet(
                ncpus=self.inferred_args.num_cpus, demos_dir=self.inferred_args.input_dataset_path
            )
        file_contents += run_closed_loop_snippet(
            app_args=self.app_args, inferred_args=self.inferred_args
        )
        file_contents += publish_closed_loop_eval_snippet()
        return file_contents

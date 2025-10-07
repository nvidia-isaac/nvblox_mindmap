# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import sys

from mindmap_osmo.tasks.closed_loop_task import ClosedLoopTask
from mindmap_osmo.tasks.training_task import TrainingTask
from mindmap_osmo.workflow_utils.arg_parsing import get_app_args
from mindmap_osmo.workflow_utils.workflow import Workflow
from mindmap_osmo.workflow_utils.workflow_args import get_workflow_args
from mindmap_osmo.workflow_utils.workflow_types import OsmoTaskType, OsmoWorkflowType


def main(cli_args: list[str]):
    # Get args
    workflow_args, non_workflow_args = get_workflow_args(
        osmo_workflow_type=OsmoWorkflowType.TRAIN_AND_EVAL, input_args=cli_args
    )
    training_app_args = get_app_args(
        osmo_workflow_type=OsmoWorkflowType.TRAIN_AND_EVAL,
        osmo_task_type=OsmoTaskType.TRAINING,
        workflow_args=workflow_args,
        input_args=non_workflow_args,
    )
    closed_loop_app_args = get_app_args(
        osmo_workflow_type=OsmoWorkflowType.TRAIN_AND_EVAL,
        osmo_task_type=OsmoTaskType.EVAL,
        workflow_args=workflow_args,
        input_args=non_workflow_args,
    )
    common_extra_args = set(training_app_args.extra_args) & set(closed_loop_app_args.extra_args)
    assert len(common_extra_args) == 0, f"Unknown arguments: {common_extra_args}"

    # Create workflow
    workflow = Workflow(
        osmo_workflow_type=OsmoWorkflowType.TRAIN_AND_EVAL,
        workflow_args=workflow_args,
        task_cls_list=[TrainingTask, ClosedLoopTask],
        app_args_list=[training_app_args, closed_loop_app_args],
    )
    workflow.generate_workflow()
    workflow.submit_workflow(dry_run=workflow_args.dry_run)


if __name__ == "__main__":
    main(sys.argv[1:])

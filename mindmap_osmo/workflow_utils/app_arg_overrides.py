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
from dataclasses import dataclass
import sys

from tap import Tap

from mindmap.image_processing.feature_extraction import FeatureExtractorType
from mindmap_osmo.workflow_utils.inferred_args import get_feature_name_for_wandb, get_num_cams_str
from mindmap_osmo.workflow_utils.workflow_constants import (
    DATA_TYPE_TO_DATASET_NAME,
    TASK_TYPE_TO_DATAGEN_DEMO_RANGES,
    TASK_TYPE_TO_DATASET_NAME,
    TASK_TYPE_TO_EVALUATION_DEMO_RANGES,
    TASK_TYPE_TO_TRAINING_DEMO_RANGES,
    TASK_TYPE_TO_TRAINING_VALIDATION_DEMO_RANGES,
    WORKFLOW_TYPE_TO_WANDB_PREFIX,
)
from mindmap_osmo.workflow_utils.workflow_types import OsmoTaskType, OsmoWorkflowType


@dataclass
class AppArgOverrides:
    """
    Dataclass containing application argument overrides.
    Arguments defined here are used to override the default values in the application arguments.
    """

    wandb_name: str
    train_iters: int
    batch_size: int
    batch_size_val: int
    demos_datagen: str
    demos_train: str
    demos_valset: str
    demos_closed_loop: str
    base_log_dir: str = "{{output}}/train_logs"
    eval_file_path: str = "{{output}}/closed_loop_eval.json"
    val_freq: int = 2500
    num_workers: int = 20
    num_workers_for_test_dataset: int = 0
    print_timers_freq: int = 100


def get_app_arg_overrides(
    workflow_type: OsmoWorkflowType,
    osmo_task_type: OsmoTaskType,
    workflow_args: argparse.Namespace,
    app_args: Tap,
) -> AppArgOverrides:
    """Get application argument overrides from application arguments and workflow type."""
    return AppArgOverrides(
        wandb_name=get_wandb_name(workflow_type, workflow_args, app_args),
        train_iters=get_train_iters(workflow_type),
        demos_datagen=get_demos_datagen(app_args.task.name),
        demos_train=get_demos_train(app_args.task.name),
        demos_valset=get_demos_valset(app_args.task.name),
        demos_closed_loop=get_demos_closed_loop(app_args.task.name),
        batch_size=get_batch_size(app_args.feature_type),
        batch_size_val=get_batch_size(app_args.feature_type),
    )


def get_demos_datagen(osmo_task_name: str) -> str:
    """Get demos based on osmo task type."""
    return TASK_TYPE_TO_DATAGEN_DEMO_RANGES[osmo_task_name]


def get_demos_train(osmo_task_name: str) -> str:
    """Get demos based on osmo task type."""
    return TASK_TYPE_TO_TRAINING_DEMO_RANGES[osmo_task_name]


def get_demos_valset(osmo_task_name: str) -> str:
    """Get demos based on osmo task type."""
    return TASK_TYPE_TO_TRAINING_VALIDATION_DEMO_RANGES[osmo_task_name]


def get_demos_closed_loop(osmo_task_name: str) -> str:
    """Get demos based on osmo task type."""
    return TASK_TYPE_TO_EVALUATION_DEMO_RANGES[osmo_task_name]


def override_app_args(
    workflow_type: OsmoWorkflowType,
    osmo_task_type: OsmoTaskType,
    workflow_args: argparse.Namespace,
    app_args: Tap,
) -> Tap:
    """Override the application arguments which are not passed on the command line with AppArgOverrides."""
    app_arg_overrides = get_app_arg_overrides(
        workflow_type, osmo_task_type, workflow_args, app_args
    )
    for arg_name, override_value in vars(app_arg_overrides).items():
        if not arg_passed_on_cli(arg_name):
            if hasattr(app_args, arg_name):
                setattr(app_args, arg_name, override_value)
    return app_args


def get_wandb_name(
    osmo_workflow_type: OsmoWorkflowType,
    workflow_args: argparse.Namespace,
    app_args: Tap,
) -> str:
    """Get wandb name based on workflow type and application arguments."""
    osmo_workflow_type_name = WORKFLOW_TYPE_TO_WANDB_PREFIX[osmo_workflow_type.name]
    task_name = TASK_TYPE_TO_DATASET_NAME[app_args.task.name]
    data_name = DATA_TYPE_TO_DATASET_NAME[app_args.data_type.name]
    feature_name = get_feature_name_for_wandb(app_args.data_type, osmo_workflow_type, app_args)
    num_cams_str = get_num_cams_str(app_args)

    wandb_name = f"{osmo_workflow_type_name}_{task_name}_{data_name}_{feature_name}_{num_cams_str}"
    if workflow_args.prefix:
        wandb_name = f"{workflow_args.prefix}_{wandb_name}"
    return wandb_name


def get_train_iters(workflow_type: OsmoWorkflowType) -> int:
    """Get train iters based on workflow type."""
    if workflow_type in [OsmoWorkflowType.TRAIN_AND_EVAL, OsmoWorkflowType.E2E]:
        # 150k iteration before starting evaluation
        return int(1.5 * 1e5)
    else:
        # ~infinite iterations (download checkpoint whenever)
        return int(1e6)


def get_batch_size(feature_type: FeatureExtractorType):
    """Get batch size based on feature type"""
    # Radio performs notoriously bad on larger batch sizes
    if feature_type == FeatureExtractorType.RADIO_V25_B:
        return 32
    else:
        return 64


def arg_passed_on_cli(arg: str) -> bool:
    """Check if the argument was passed on the command line."""
    return f"--{arg}" in sys.argv[1:]

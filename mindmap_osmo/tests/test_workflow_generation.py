# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import List

from mindmap.data_loading.data_types import DataType
from mindmap.image_processing.feature_extraction import FeatureExtractorType
from mindmap.tasks.tasks import Tasks
from mindmap_osmo.submit_datagen_workflow import main as run_datagen_workflow
from mindmap_osmo.submit_e2e_workflow import main as run_e2e_workflow
from mindmap_osmo.submit_evaluation_workflow import main as run_evaluation_workflow
from mindmap_osmo.submit_train_and_eval_workflow import main as run_train_and_eval_workflow
from mindmap_osmo.submit_training_workflow import main as run_training_workflow
from mindmap_osmo.workflow_utils.workflow_constants import PlatformType

FEATURE_TYPES_TO_TEST = [FeatureExtractorType.RADIO_V25_B, FeatureExtractorType.CLIP_RESNET50_FPN]
TASKS_TO_TEST = [Tasks.CUBE_STACKING, Tasks.MUG_IN_DRAWER]
DATA_TYPES_TO_TEST = [DataType.RGBD, DataType.MESH, DataType.RGBD_AND_MESH]


def _get_arg_permutations(
    checkpoint: str = None,
    platform: PlatformType = None,
    feature_types: List[FeatureExtractorType] = [None],
    tasks: List[Tasks] = [None],
    data_types: List[DataType] = [None],
) -> List[List[str]]:
    """Generate argument permutations for workflow testing.

    Args:
        checkpoint (str, optional): Path to checkpoint file. Defaults to None.
        platform (PlatformType, optional): Platform type to run on. Defaults to None.
        feature_types (List[FeatureExtractorType], optional): List of feature extractor types to test. Defaults to [None].
        tasks (List[Tasks], optional): List of tasks to test. Defaults to [None].
        data_types (List[DataType], optional): List of data types to test. Defaults to [None].

    Returns:
        List[List[str]]: List of argument lists, where each inner list contains CLI-style arguments
        for a specific permutation of the input parameters.
    """
    non_cli_args = []
    for feature_type in feature_types:
        for task in tasks:
            for data_type in data_types:
                args = []
                if feature_type:
                    args.extend(["--feature_type", feature_type.value])
                if task:
                    args.extend(["--task", task.value])
                if data_type:
                    args.extend(["--data_type", data_type.value])
                if checkpoint:
                    args.extend(["--checkpoint", checkpoint])
                if platform:
                    args.extend(["--platform", platform.value])
                args.extend(["--dry-run"])
                non_cli_args.append(args)
    return non_cli_args


def test_e2e_workflow():
    arg_permutations = _get_arg_permutations(
        feature_types=FEATURE_TYPES_TO_TEST,
        tasks=TASKS_TO_TEST,
        data_types=DATA_TYPES_TO_TEST,
    )
    for args in arg_permutations:
        print(f"Generating e2e workflow with args: {args}")
        run_e2e_workflow(cli_args=args)


def test_train_and_eval_workflow():
    arg_permutations = _get_arg_permutations(
        feature_types=FEATURE_TYPES_TO_TEST,
        tasks=TASKS_TO_TEST,
        data_types=DATA_TYPES_TO_TEST,
    )
    for args in arg_permutations:
        print(f"Generating train and eval workflow with args: {args}")
        run_train_and_eval_workflow(cli_args=args)


def test_closed_loop_workflow():
    arg_permutations = _get_arg_permutations(
        checkpoint="checkpoint.pth",
        feature_types=FEATURE_TYPES_TO_TEST,
        tasks=TASKS_TO_TEST,
        data_types=DATA_TYPES_TO_TEST,
    )
    for args in arg_permutations:
        print(f"Generating closed loop workflow with args: {args}")
        run_evaluation_workflow(cli_args=args)


def test_datagen_workflow():
    arg_permutations = _get_arg_permutations(
        feature_types=FEATURE_TYPES_TO_TEST,
        tasks=TASKS_TO_TEST,
        data_types=DATA_TYPES_TO_TEST,
    )
    for args in arg_permutations:
        print(f"Generating datagen workflow with args: {args}")
        run_datagen_workflow(cli_args=args)


def test_training_workflow():
    arg_permutations = _get_arg_permutations(
        platform=PlatformType.OVX_L40,
        feature_types=FEATURE_TYPES_TO_TEST,
        tasks=TASKS_TO_TEST,
        data_types=DATA_TYPES_TO_TEST,
    )
    for args in arg_permutations:
        print(f"Generating training workflow with args: {args}")
        run_training_workflow(cli_args=args)

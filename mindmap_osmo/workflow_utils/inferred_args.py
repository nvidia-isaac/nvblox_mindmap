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
import math
import os
from typing import Dict, Optional, Tuple

from tap import Tap

from mindmap.closed_loop.closed_loop_mode import ClosedLoopMode
from mindmap.common_utils.demo_selection import get_demo_super_range, get_num_demos
from mindmap.data_loading.data_types import DataType, includes_nvblox, includes_rgb
from mindmap.embodiments.embodiment_base import EmbodimentType
from mindmap.embodiments.task_to_embodiment import get_embodiment_type_from_task
from mindmap.image_processing.feature_extraction import FeatureExtractorType
from mindmap.tasks.tasks import Tasks
from mindmap_osmo.workflow_utils.workflow_constants import (
    CHECKPOINT_DATASET_NAME,
    DATA_TYPE_TO_DATASET_NAME,
    DEFAULT_MEMORY_GB,
    DEFAULT_NUM_CPUS,
    DEFAULT_NUM_GPUS,
    DEMO_SIZE_GB,
    EVAL_STORAGE_GB,
    FEATURE_TYPE_TO_DATASET_NAME,
    HDF5_DATASET_NAME,
    MIN_STORAGE_GB,
    TASK_TYPE_TO_DATASET_NAME,
    TASK_TYPE_TO_FPN_FILE_NAME,
    TASK_TYPE_TO_HDF5_FILE_NAME,
)
from mindmap_osmo.workflow_utils.workflow_types import DatasetType, OsmoTaskType, OsmoWorkflowType


@dataclass
class InferredArgs:
    """Dataclass containing arguments inferred from application and workflow arguments/types.
    These arguments are used to define an osmo task as part of a workflow.

    Attributes:
        num_cpus: Number of CPUs to use
        num_gpus: Number of GPUs to use
        memory_gb: Memory to use in GB
        storage_gb: Storage to use in GB
        hdf5_file_name: Name of the HDF5 file
        fpn_file_name: Name of the FPN checkpoint file
        fpn_osmo_path: Full path to the FPN checkpoint in OSMO
        hdf5_osmo_path: Full path to the HDF5 file in OSMO
        checkpoint_osmo_path: Full path to the model checkpoint in OSMO
        input_dataset_name: Name of the input dataset (optional)
        input_dataset_regex: Regex pattern to match input demo files (optional)
        input_dataset_path: Full path to the input dataset in OSMO
        output_dataset_name: Name of the output dataset (optional)
        output_dataset_regex: Regex pattern to match output demo files (optional)
    """

    num_cpus: int
    num_gpus: int
    memory_gb: int
    storage_gb: int
    hdf5_file_name: str
    fpn_file_name: str
    fpn_osmo_path: str
    hdf5_osmo_path: str
    checkpoint_osmo_path: str
    input_dataset_name: str
    input_dataset_regex: str
    input_dataset_path: str
    output_dataset_name: str
    output_dataset_regex: str


def get_inferred_args(
    osmo_workflow_type: OsmoWorkflowType,
    workflow_args: argparse.Namespace,
    osmo_task_type: OsmoTaskType,
    app_args: Tap,
    dataset_indices: Dict[str, int],
) -> InferredArgs:
    """Get inferred arguments from application arguments."""
    # Command line overrides.
    output_dataset_name = get_dataset_name(
        data_type=app_args.data_type,
        app_args=app_args,
        dataset_name_override=workflow_args.output_dataset_name,
    )
    input_dataset_name = get_dataset_name(
        data_type=app_args.data_type,
        app_args=app_args,
        dataset_name_override=workflow_args.input_dataset_name,
        version=workflow_args.input_dataset_version,
    )

    demo_range = get_demo_range(osmo_task_type, app_args)
    return InferredArgs(
        num_cpus=get_default_num_cpus(osmo_task_type),
        num_gpus=get_default_num_gpus(osmo_task_type),
        memory_gb=get_default_memory(osmo_task_type),
        storage_gb=get_default_storage(osmo_task_type, app_args),
        hdf5_file_name=get_hdf5_file_name(app_args.task),
        fpn_file_name=get_fpn_file_name(app_args.task, osmo_task_type, app_args),
        fpn_osmo_path=get_fpn_osmo_path(app_args.task, osmo_task_type, app_args, dataset_indices),
        hdf5_osmo_path=get_hdf5_osmo_path(app_args.task, dataset_indices),
        checkpoint_osmo_path=get_checkpoint_osmo_path(
            osmo_workflow_type, osmo_task_type, app_args, workflow_args, dataset_indices
        ),
        input_dataset_name=input_dataset_name,
        input_dataset_regex=get_dataset_regex(demo_range),
        input_dataset_path=get_dataset_path(
            osmo_workflow_type, osmo_task_type, input_dataset_name, dataset_indices
        ),
        output_dataset_name=output_dataset_name,
        output_dataset_regex=get_dataset_regex(demo_range),
    )


def get_default_memory(osmo_task_type: OsmoTaskType) -> int:
    """Get default memory for given osmo task type (int in GB)."""
    assert osmo_task_type.name in DEFAULT_MEMORY_GB, f"Invalid osmo task type: {osmo_task_type}"
    return DEFAULT_MEMORY_GB[osmo_task_type.name]


def get_demo_range(osmo_task_type: OsmoTaskType, app_args: Tap) -> str:
    """Get the demo range for the given osmo task type."""
    if osmo_task_type == OsmoTaskType.EVAL:
        return app_args.demos_closed_loop
    elif osmo_task_type == OsmoTaskType.DATAGEN:
        return app_args.demos_datagen
    elif osmo_task_type in [OsmoTaskType.TRAINING]:
        return get_demo_super_range(app_args.demos_train, app_args.demos_valset)
    else:
        raise ValueError(f"Invalid osmo task type: {osmo_task_type}")


def get_default_storage(osmo_task_type: OsmoTaskType, app_args: Tap) -> int:
    """Get default storage for given osmo task type (int in GB)."""
    if (
        osmo_task_type == OsmoTaskType.EVAL
        and app_args.demo_mode != ClosedLoopMode.EXECUTE_GT_GOALS
    ):
        return EVAL_STORAGE_GB
    else:
        # Find the dataset that will be generated/loaded in this workflow.
        dataset_name = get_dataset_name(app_args.data_type, app_args)
        # Calculate the storage required for the dataset.
        assert dataset_name in DEMO_SIZE_GB, f"Dataset {dataset_name} not found in DEMO_SIZE_GB"

        demo_range = get_demo_range(osmo_task_type, app_args)
        num_demos = get_num_demos(demo_range)
        assert num_demos > 0, f"Number of demos must be greater than 0, got {num_demos}"
        dataset_size_gb = math.ceil(DEMO_SIZE_GB[dataset_name] * num_demos)
        return max(dataset_size_gb, MIN_STORAGE_GB)


def get_default_num_gpus(osmo_task_type: OsmoTaskType) -> int:
    """Get default number of GPUs for given osmo task type."""
    assert osmo_task_type.name in DEFAULT_NUM_GPUS, f"Invalid osmo task type: {osmo_task_type}"
    return DEFAULT_NUM_GPUS[osmo_task_type.name]


def get_default_num_cpus(osmo_task_type: OsmoTaskType) -> int:
    """Get default number of CPUs for given osmo task type."""
    assert osmo_task_type.name in DEFAULT_NUM_CPUS, f"Invalid osmo task type: {osmo_task_type}"
    return DEFAULT_NUM_CPUS[osmo_task_type.name]


def get_num_cams_str(app_args: Tap) -> str:
    """Get the number of cameras for the wandb name."""
    return f"{2 if app_args.add_external_cam else 1}cam"


def get_feature_name_for_dataset(data_type: DataType, app_args: Tap) -> str:
    """Get the feature name for the dataset name."""
    # NOTE(alex): Difference to below: in the dataset case + RGBD we always load from x-featured
    #             datasets, even if training with e.g. radio. In the wandb name case, we label the
    #             the correct feature in the name, except in the DATAGEN+RGBD case,
    #             we always label as x-featured.
    if data_type == DataType.RGBD:
        return "x"
    else:
        return FEATURE_TYPE_TO_DATASET_NAME[app_args.feature_type.name]


def get_feature_name_for_wandb(
    data_type: DataType, workflow_type: OsmoWorkflowType, app_args: Tap
) -> str:
    """Get the feature name for the wandb name."""
    # NOTE(alex): Difference to below: in the dataset case + RGBD we always load from x-featured
    #             datasets, even if training with e.g. radio. In the wandb name case, we label the
    #             the correct feature in the name, except in the DATAGEN+RGBD case,
    #             we always label as x-featured.
    if workflow_type == OsmoWorkflowType.DATAGEN and data_type == DataType.RGBD:
        return "x"
    else:
        return FEATURE_TYPE_TO_DATASET_NAME[app_args.feature_type.name]


def get_dataset_name(
    data_type: DataType,
    app_args: Tap,
    dataset_name_override: Optional[str] = None,
    version: Optional[int] = None,
) -> str:
    """Get the name of the dataset based on the task, data type, feature type, and wrist cam only flag."""
    # Automatically generate the dataset name.
    if dataset_name_override is None:
        # Get the task and data name.
        task_name = TASK_TYPE_TO_DATASET_NAME[app_args.task.name]

        # Get the data name.
        data_name = DATA_TYPE_TO_DATASET_NAME[data_type.name]

        # Get the feature name
        feature_name = get_feature_name_for_dataset(data_type, app_args)

        # Get the number of cameras string
        num_cams_str = get_num_cams_str(app_args)

        # Return the dataset name.
        dataset_name = f"mindmap_{task_name}_{data_name}_{feature_name}_{num_cams_str}"
    # Dataset name override provided.
    else:
        dataset_name = dataset_name_override

    # Add the version if requested.
    if version is not None:
        dataset_name += f":{version}"

    # Return the dataset name.
    return dataset_name


def get_dataset_regex(demos: str, demos_valset: Optional[str] = None) -> str:
    """Get the regex for the dataset based on the demo range (useful for selective downloading)."""

    # Combine the training and valset ranges
    if demos_valset is not None:
        demos = get_demo_super_range(demos, demos_valset)

    # Parse the range of demos.
    if "-" in demos:
        # Parse range of demos.
        first_demo, last_demo = map(int, demos.split("-"))
        assert first_demo >= 0 and first_demo <= 99999
        assert last_demo >= 0 and last_demo <= 99999
        assert first_demo <= last_demo

        # Parse the last demo to get the number of leading zeros and the highest digit.
        last_demo_str = str(abs(last_demo))
        number_of_digits = len(last_demo_str)
        number_of_leading_zeros = 5 - number_of_digits
        highest_digit = last_demo_str[0]

        # Build the regex.
        # NOTE(remos): Our generated regex matches a superset of the selected demo range.
        # E.g. for the range 11-121, the generated regex will be .*/?demo_000[0-1][0-9][0-9].
        # To match the selected range exactly, we would need to match subranges (using the | operator),
        # which is omitted for simplicity and time reasons.
        regex = f".*/?demo_" + "0" * number_of_leading_zeros
        regex += f"([0-{highest_digit}]"
        regex += f"[0-9]" * (number_of_digits - 1) + ")"
    else:
        assert int(demos) >= 0 and int(demos) <= 99999
        regex = f"demo_{int(demos):05d}"

    return regex


def get_hdf5_file_name(task: Tasks) -> str:
    """Get the name of the HDF5 file for the given task."""
    assert (
        task.name in TASK_TYPE_TO_HDF5_FILE_NAME
    ), f"Task {task.name} not implemented in get_hdf5_file_name"
    return TASK_TYPE_TO_HDF5_FILE_NAME[task.name]


def remove_version_number(dataset_name: str) -> str:
    """Remove the version number from the dataset name if it has one."""
    return dataset_name.split(":")[0]


def get_dataset_path(
    osmo_workflow_type: OsmoWorkflowType,
    osmo_task_type: OsmoTaskType,
    dataset_name: str,
    dataset_indices: Dict[str, int],
) -> str:
    """Get the path to the dataset based on the workflow type and application arguments."""
    if osmo_workflow_type == OsmoWorkflowType.E2E and osmo_task_type == OsmoTaskType.TRAINING:
        # For E2E training, we use dataset generated by the datagen task as input to the training task.
        task_dataset_idx = dataset_indices[DatasetType.TASK.name]
        dataset_path = f"{{{{input:{task_dataset_idx}}}}}"
    else:
        dataset_idx = dataset_indices[DatasetType.DATA.name]
        if dataset_idx is None:
            return None
        dataset_name = remove_version_number(dataset_name)
        dataset_path = f"{{{{input:{dataset_idx}}}}}/{dataset_name}"
    return dataset_path


def get_fpn_osmo_path(
    task: Tasks, osmo_task_type: OsmoTaskType, app_args: Tap, dataset_indices: Dict[str, int]
) -> str:
    """Get the path to the FPN checkpoint in OSMO."""
    checkpoint_dataset_idx = dataset_indices[DatasetType.CHECKPOINT.name]
    if checkpoint_dataset_idx is None:
        return None
    fpn_file_name = get_fpn_file_name(task, osmo_task_type, app_args)
    if fpn_file_name is None:
        return None
    return f"{{{{input:{checkpoint_dataset_idx}}}}}/{CHECKPOINT_DATASET_NAME}/{fpn_file_name}"


def get_hdf5_osmo_path(task: Tasks, dataset_indices: Dict[str, int]) -> str:
    """Get the path to the HDF5 file in OSMO."""
    hdf5_dataset_idx = dataset_indices[DatasetType.HDF5.name]
    if hdf5_dataset_idx is None:
        return None
    return f"{{{{input:{hdf5_dataset_idx}}}}}/{HDF5_DATASET_NAME}/{get_hdf5_file_name(task)}"


def get_checkpoint_osmo_path(
    osmo_workflow_type: OsmoWorkflowType,
    osmo_task_type: OsmoTaskType,
    app_args: Tap,
    workflow_args: argparse.Namespace,
    dataset_indices: Dict[str, int],
) -> str:
    """Get the path to the checkpoint file in OSMO."""
    if app_args.checkpoint is None:
        assert not checkpoint_arg_required(
            osmo_workflow_type, app_args
        ), "Please pass a checkpoint file"
        if use_checkpoint_from_training_task(osmo_workflow_type, osmo_task_type):
            task_dataset_idx = dataset_indices[DatasetType.TASK.name]
            path = f"{{{{input:{task_dataset_idx}}}}}/train_logs/checkpoints/*/best.pth"
            checkpoint_path = f"$(python3 -c \"import glob; print(glob.glob('{path}')[0])\")"
        else:
            return None
    else:
        if workflow_args.checkpoint_from_swiftstack:
            checkpoint_dataset_idx = dataset_indices[DatasetType.CHECKPOINT_SWIFT.name]
            # In swiftstack, the dataset input already contains the whole path, so we just need to
            # extract the filename.
            checkpoint_path = (
                f"{{{{input:{checkpoint_dataset_idx}}}}}/{os.path.split(app_args.checkpoint)[-1]}"
            )
        else:
            checkpoint_dataset_idx = dataset_indices[DatasetType.CHECKPOINT.name]
            checkpoint_path = f"{{{{input:{checkpoint_dataset_idx}}}}}/{CHECKPOINT_DATASET_NAME}/{app_args.checkpoint}"

        assert checkpoint_dataset_idx is not None, "Checkpoint dataset index is not set"

    return checkpoint_path


def get_fpn_file_name(task: Tasks, osmo_task_type: OsmoTaskType, app_args: Tap) -> str:
    """Get the name of the FPN checkpoint file for the given task."""
    assert (
        task.name in TASK_TYPE_TO_FPN_FILE_NAME
    ), f"Task {task.name} not implemented in get_fpn_file_name"
    if fpn_required(osmo_task_type, app_args):
        return TASK_TYPE_TO_FPN_FILE_NAME[task.name]
    else:
        return None


def use_checkpoint_from_training_task(
    osmo_workflow_type: OsmoWorkflowType, osmo_task_type: OsmoTaskType
) -> bool:
    """Check if we should use the checkpoint from the training task for the given workflow type (instead of using a checkpoint from a dataset)."""
    if osmo_workflow_type in [OsmoWorkflowType.E2E, OsmoWorkflowType.TRAIN_AND_EVAL]:
        return osmo_task_type == OsmoTaskType.EVAL
    else:
        return False


def checkpoint_arg_required(osmo_workflow_type: OsmoWorkflowType, app_args: Tap) -> bool:
    """Check if a checkpoint is required for the given workflow type and application arguments."""
    if osmo_workflow_type == OsmoWorkflowType.EVAL:
        return app_args.demo_mode != ClosedLoopMode.EXECUTE_GT_GOALS
    else:
        return False


def fpn_required(osmo_task_type: OsmoTaskType, app_args: Tap) -> bool:
    """Check if a FPN checkpoint is required for the given workflow type and application arguments."""
    if app_args.feature_type == FeatureExtractorType.CLIP_RESNET50_FPN:
        if osmo_task_type == OsmoTaskType.DATAGEN:
            # Required for generating features for mapping.
            return includes_nvblox(app_args.data_type)
        elif osmo_task_type == OsmoTaskType.TRAINING:
            # In the nvblox case, the FPN is not needed (built into the reconstruction).
            # In the RGBD case, the FPN is trained along with the rest of the model.
            # In the RGBD_AND_MESH case, the FPN needs to be loaded to correspond to the one used for mapping.
            return app_args.data_type == DataType.RGBD_AND_MESH
        elif osmo_task_type == OsmoTaskType.EVAL:
            # Required for generating features for mapping.
            # In the RGBD case, the FPN is loaded along with the rest of the model.
            return includes_nvblox(app_args.data_type)
        else:
            raise ValueError(f"Invalid osmo task type: {osmo_task_type}")
    else:
        return False

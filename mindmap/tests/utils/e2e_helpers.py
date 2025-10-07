# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import glob
import importlib
import os
import random
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional

import torch

from mindmap.common_utils.system import get_random_port_in_unassigned_range
from mindmap.data_loading.data_types import DataType
from mindmap.data_loading.dataset import SamplingWeightingType
from mindmap.data_loading.dataset_files_by_encoding_method import get_data_loader_by_data_type
from mindmap.data_loading.item_names import (
    GT_POLICY_STATE_PRED_ITEM_NAME,
    IS_KEYPOSE_ITEM_NAME,
    POLICY_STATE_HISTORY_ITEM_NAME,
)
from mindmap.data_loading.vertex_sampling import VertexSamplingMethod
from mindmap.embodiments.arm.embodiment import ArmEmbodiment
from mindmap.embodiments.embodiment_base import EmbodimentBase
from mindmap.embodiments.humanoid.observation import (
    EXTERNAL_DEPTH_ITEM_NAME,
    EXTERNAL_RGB_ITEM_NAME,
    POV_DEPTH_ITEM_NAME,
    POV_RGB_ITEM_NAME,
)
from mindmap.embodiments.humanoid.policy_state import HumanoidEmbodimentPolicyState
from mindmap.isaaclab_utils.isaaclab_datagen_utils import (
    DemoOutcome,
    demo_directory_from_episode_name,
)
from mindmap.isaaclab_utils.isaaclab_writer import IsaacLabWriter
from mindmap.keyposes.keypose_detection_mode import KeyposeDetectionMode
from mindmap.keyposes.task_to_default_keypose_params import TASK_TYPE_TO_KEYPOSE_DETECTION_MODE
from mindmap.tasks.tasks import Tasks
from mindmap.tests.utils.comparisons import datasets_are_close
from mindmap.tests.utils.constants import TestDataLocations


def setup_datagen(output_dir: str) -> dict:
    # Set the seed to make the dataset generation deterministic
    torch.manual_seed(0)

    # Wipe the output dir.
    shutil.rmtree(output_dir, ignore_errors=True)

    # Headless needed
    env = os.environ.copy()
    env["HEADLESS"] = "1"

    return env


def run_datagen(
    output_dir: str,
    task: Tasks,
    data_type: DataType,
    add_external_cam: bool,
    hdf5_filepath: str,
    env: dict,
) -> None:
    DATAGEN_SCRIPT_PATH = f"{TestDataLocations.repo_root}/mindmap/run_isaaclab_datagen.py"
    run_subprocess(
        [
            "python3",
            DATAGEN_SCRIPT_PATH,
            "--num_envs",
            "1",
            "--task",
            task.value,
            "--hdf5_file",
            hdf5_filepath,
            "--output_dir",
            output_dir,
            "--data_type",
            data_type.value,
            "--feature_type",
            "rgb",  # Using RGB feature since (1) CLIP+fpn is non-deterministic and (2) RADIO is prohibitively large to store in gitlfs
            "--demos_datagen",
            "0",
            "--max_num_attempts",
            "1",
            "--max_num_steps",
            "50",
            "--render_settings",
            "deterministic",
        ]
        + (["--add_external_cam"] if add_external_cam else []),
        env=env,
    )


def replace_baseline_dataset(generated_output_dir: str, baseline_output_dir: str) -> None:
    os.makedirs(baseline_output_dir, exist_ok=True)
    run_subprocess(
        [
            "python3",
            f"{TestDataLocations.repo_root}/mindmap/scripts/tar_demos.py",
            "--demos_dir",
            generated_output_dir,
            "--output_dir",
            baseline_output_dir,
        ]
    )


def dataset_equal_to_baseline(
    embodiment: EmbodimentBase,
    task: Tasks,
    demo_idx: int,
    generated_dataset_dir: str,
    baseline_dataset_dir: str,
    data_type: DataType,
    add_external_cam: bool,
    batch_size: int,
    vertex_sampling_method: VertexSamplingMethod,
    num_batches_to_compare: int = None,
    apply_geometry_noise_to_baseline: bool = False,
) -> bool:
    """Compare the generated dataset with stored baseline. We expect them to be equal"""
    # Untar the baseline dataset
    run_subprocess(
        [
            "python3",
            f"{TestDataLocations.repo_root}/mindmap/scripts/untar_demos.py",
            "--demos_dir",
            baseline_dataset_dir,
            "--num_processes",
            "1",
        ]
    )

    dataset_new = get_dataloader(
        embodiment,
        dataset_path=generated_dataset_dir,
        demos=str(demo_idx),
        task=task,
        data_type=data_type,
        batch_size=batch_size,
        vertex_sampling_method=vertex_sampling_method,
        add_external_cam=add_external_cam,
    )
    dataset_gt = get_dataloader(
        embodiment,
        dataset_path=baseline_dataset_dir,
        demos=str(demo_idx),
        task=task,
        data_type=data_type,
        batch_size=batch_size,
        apply_geometry_noise=apply_geometry_noise_to_baseline,
        vertex_sampling_method=vertex_sampling_method,
        add_external_cam=add_external_cam,
    )
    return datasets_are_close(
        embodiment,
        dataset_new,
        dataset_gt,
        batch_size=batch_size,
        data_type=data_type,
        verbose=True,
        num_batches_to_compare=num_batches_to_compare,
    )


def test_validate_demos(dataset_path: str, task: Tasks, data_type: DataType):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy the dataset to a temporary directory and set outcome to success
        shutil.copytree(dataset_path, tmp_dir, dirs_exist_ok=True)
        demo_directory = demo_directory_from_episode_name(tmp_dir, "demo_00000")
        IsaacLabWriter(demo_directory).write_outcome(DemoOutcome.SUCCESS)

        # Demo should be successful before validation
        data_loader_before = get_dataloader(
            embodiment=ArmEmbodiment(),
            dataset_path=tmp_dir,
            demos="0",
            task=task,
            include_failed_demos=True,
            data_type=data_type,
        )
        assert data_loader_before.dataset.is_demo_successful(
            demo_directory
        ), "Demo should be successful before validation"

        # Run the validation. Outcome should be "fail" since the demo does not have not enough frames to successfully complete the task.
        run_subprocess(
            [
                "python3",
                f"{TestDataLocations.repo_root}/mindmap/run_validate_demos.py",
                "--headless",
                "--dataset",
                tmp_dir,
                "--task",
                task.value,
                "--data_type",
                data_type.value,
                "--hdf5_file",
                f"{TestDataLocations.cube_stacking_hdf5_filepath}",
            ]
        )

        # Demo should be unsuccessful after validation
        data_loader = get_dataloader(
            embodiment=ArmEmbodiment(),
            dataset_path=tmp_dir,
            demos="0",
            task=task,
            include_failed_demos=True,
            data_type=data_type,
        )

        assert not data_loader.dataset.is_demo_successful(
            demo_directory
        ), "Demo should be unsuccessful after validation"


def test_closed_loop(
    checkpoint_filepath: str,
    task: Tasks,
    data_type: DataType,
    add_external_cam: bool,
    hdf5_filepath: str,
    additional_args: Optional[List[str]] = None,
):
    # Get the closed-loop script path
    module_name = "mindmap.run_closed_loop_policy"
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"Cannot find the module: {module_name}")
    script_path = spec.origin
    print(f"Found closed-loop script at: {script_path}")

    # Find the model we produced from the training step
    assert os.path.exists(
        checkpoint_filepath
    ), f"The passed checkpoint file doesnt exist: {checkpoint_filepath}"
    assert os.path.exists(
        checkpoint_filepath
    ), f"You need a checkpoint file at {checkpoint_filepath} to run this test."

    # Run the closed-loop.
    # TODO(alexmillane): We have to run through subprocess, rather than just calling a function,
    # because of our current requirement to run through 'torchrun'. When we get rid of this
    # requirement, move to testing through a function call, rather than through subprocess.
    NUM_STEPS = 10
    TIMEOUT_S = 240

    with tempfile.TemporaryDirectory() as tmp_dir:
        eval_file_path = os.path.join(tmp_dir, "eval.json")
        video_path = os.path.join(tmp_dir, "video")
        html_output_path = os.path.join(tmp_dir, "html")
        try:
            result = subprocess.run(
                ["torchrun"]
                + [
                    "--nnodes=1",
                    "--nproc_per_node=1",
                    "--master_port",
                    f"{get_random_port_in_unassigned_range()}",
                ]
                + [str(script_path)]
                + [
                    "--headless",
                    "--num_envs",
                    "1",
                    "--demos_closed_loop",
                    "0",
                    "--hdf5_file",
                    hdf5_filepath,
                    "--checkpoint",
                    f"{checkpoint_filepath}",
                    "--demo_mode",
                    "closed_loop_wait",
                    "--use_keyposes",
                    "1",
                    "--wandb_mode",
                    "offline",
                    "--terminate_after_n_steps",
                    f"{NUM_STEPS}",
                    "--record_camera_output_path",
                    video_path,
                    "--record_video",
                    "--eval_file_path",
                    eval_file_path,
                    "--task",
                    task.value,
                    "--data_type",
                    data_type.value,
                ]
                + (["--add_external_cam"] if add_external_cam else [])
                + (additional_args or []),
                check=True,
                timeout=TIMEOUT_S,
            )
            assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
        except subprocess.TimeoutExpired as e:
            print(f"Process timed out after {TIMEOUT_S} seconds")
            raise

        assert_files_created(f"{video_path}/demo_0.mp4")

        # Generate output html
        try:
            result = subprocess.run(
                [
                    "python3",
                    f"{TestDataLocations.repo_root}/mindmap/scripts/publish_closed_loop_eval.py",
                    "--eval_file_path",
                    eval_file_path,
                    "--videos_path",
                    video_path,
                    "--output_path",
                    html_output_path,
                ],
                check=True,
                timeout=TIMEOUT_S,
            )
            assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
        except subprocess.TimeoutExpired as e:
            print(f"Process timed out after {TIMEOUT_S} seconds")
            raise

        assert_files_created(f"{html_output_path}/index.html")


def test_training(
    dataset: str,
    demo_idx: int,
    training_output_dir: str,
    task: Tasks,
    data_type: DataType,
    add_external_cam: bool,
    additional_train_args: Optional[str] = None,
):
    """Helper for running training and check that output files are created"""
    # Some training constants
    TRAIN_ITERS = 12
    NUM_TRAIN_EVAL = 2
    NUM_TEST_EVAL = 3
    VAL_FREQ = 10

    shutil.rmtree(training_output_dir, ignore_errors=True)

    TRAIN_SCRIPT_PATH = f"{TestDataLocations.repo_root}/mindmap/run_training.py"
    port = get_random_port_in_unassigned_range()
    cmd_args = [
        "torchrun",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc_per_node",
        "1",
        "--master_port",
        str(port),
        TRAIN_SCRIPT_PATH,
        "--dataset",
        dataset,
        "--demos_train",
        str(demo_idx),
        "--train_iters",
        str(TRAIN_ITERS),
        "--num_batches_per_train_eval",
        str(NUM_TRAIN_EVAL),
        "--num_batches_per_test_eval",
        str(NUM_TEST_EVAL),
        "--val_freq",
        str(VAL_FREQ),
        "--base_log_dir",
        training_output_dir,
        "--wandb_mode",
        "offline",
        "--feature_type",
        "rgb",
        "--use_keyposes",
        "1",
        "--task",
        task.value,
        "--data_type",
        data_type.value,
    ]
    if add_external_cam:
        cmd_args.append("--add_external_cam")
    if additional_train_args:
        cmd_args.extend(additional_train_args.split())
    subprocess.run(cmd_args)

    assert_files_created(f"{training_output_dir}/checkpoints/*/best.pth")
    assert_files_created(f"{training_output_dir}/checkpoints/*/last.pth")


def get_dataloader(
    embodiment: EmbodimentBase,
    dataset_path: str,
    demos: str,
    task: Tasks,
    use_keyposes=True,
    only_sample_keyposes=False,
    batch_size=16,
    gripper_encoding_mode="binary",
    num_history=3,
    prediction_horizon=1,
    add_external_cam=True,
    data_type=DataType.RGBD,
    apply_geometry_noise=False,
    vertex_sampling_method=VertexSamplingMethod.RANDOM_WITHOUT_REPLACEMENT,
    include_failed_demos=False,
):
    """Get dataloaders used during testing"""
    train_loader, _ = get_data_loader_by_data_type(
        embodiment=embodiment,
        dataset_path=dataset_path,
        demos=demos,
        task=task,
        num_workers=0,
        batch_size=batch_size,
        use_keyposes=use_keyposes,
        data_type=data_type,
        only_sample_keyposes=only_sample_keyposes,
        extra_keyposes_around_grasp_events=[],
        keypose_detection_mode=TASK_TYPE_TO_KEYPOSE_DETECTION_MODE[task.name],
        include_failed_demos=include_failed_demos,
        sampling_weighting_type=SamplingWeightingType.NONE,
        gripper_encoding_mode=gripper_encoding_mode,
        num_history=num_history,
        prediction_horizon=prediction_horizon,
        apply_random_transforms=False,
        apply_geometry_noise=apply_geometry_noise,
        pos_noise_stddev_m=1.0,
        rot_noise_stddev_deg=1.0,
        add_external_cam=add_external_cam,
        num_vertices_to_sample=1024,
        vertex_sampling_method=vertex_sampling_method,
        random_translation_range_m=None,
        random_rpy_range_deg=None,
        seed=0,
    )
    return train_loader


def validate_humanoid_batch(
    batch,
    batch_size,
    num_history,
    prediction_horizon,
    add_external_cam,
):
    """ "Assert that the dimensions of the batchs are as expected"""
    state_size = HumanoidEmbodimentPolicyState.state_size()
    assert batch[POLICY_STATE_HISTORY_ITEM_NAME].shape == (batch_size, num_history, state_size)
    assert batch[GT_POLICY_STATE_PRED_ITEM_NAME].shape == (
        batch_size,
        prediction_horizon,
        state_size,
    )
    assert batch[IS_KEYPOSE_ITEM_NAME].shape == torch.Size([3])
    assert batch[POV_RGB_ITEM_NAME].shape == (batch_size, 3, 512, 512)
    assert batch[POV_DEPTH_ITEM_NAME].shape == (batch_size, 512, 512)
    if add_external_cam:
        assert batch[EXTERNAL_RGB_ITEM_NAME].shape == (batch_size, 3, 512, 512)
        assert batch[EXTERNAL_DEPTH_ITEM_NAME].shape == (batch_size, 512, 512)
        # Check that we didn't duplicate images by accident
        assert not torch.all(batch[POV_RGB_ITEM_NAME] == batch[EXTERNAL_RGB_ITEM_NAME])
        assert not torch.all(batch[POV_DEPTH_ITEM_NAME] == batch[EXTERNAL_DEPTH_ITEM_NAME])


def run_subprocess(cmd, env=None):
    print(f"Running command: {cmd}")
    try:
        # Don't capture output, let it flow through in real-time
        result = subprocess.run(
            cmd,
            check=True,
            env=env,
            capture_output=False,
            text=True,
            # Explicitly set stdout and stderr to None to use parent process's pipes
            stdout=None,
            stderr=None,
        )
        print(f"Command completed with return code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"Command failed with return code {e.returncode}: {e}\n")
        raise


def assert_files_created(glob_str, num_expected=1, min_expected_size=0):
    """Check that globbed files exists and are nonzero"""
    try:
        paths = glob.glob(glob_str)
        assert len(paths) == num_expected
        for path in paths:
            assert os.path.getsize(path) > min_expected_size
    except:
        print(f"Expected files not created: {glob_str}")
        raise

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
import os

import pytest
import torch

from mindmap.data_loading.data_types import DataType
from mindmap.data_loading.item_names import IS_KEYPOSE_ITEM_NAME
from mindmap.data_loading.vertex_sampling import VertexSamplingMethod
from mindmap.embodiments.arm.embodiment import ArmEmbodiment
from mindmap.embodiments.humanoid.embodiment import HumanoidEmbodiment
from mindmap.tasks.tasks import Tasks
from mindmap.tests.utils.constants import TestDataLocations
import mindmap.tests.utils.e2e_helpers as helpers

"""End-to-end smoke test for datagen -> training -> closed loop"""

DEMO_IDX = 0

# Path for storing the generated dataset
FRANKA_DATASET_OUTPUT_DIR = os.path.join(TestDataLocations.generated_data_dir, "franka")
DRILL_IN_BOX_DATASET_OUTPUT_DIR = os.path.join(TestDataLocations.generated_data_dir, "drill_in_box")

# Path for baseline dataset
FRANKA_BASELINE_OUTPUT_DIR = os.path.join(TestDataLocations.baseline_data_dir, "franka")
DRILL_IN_BOX_BASELINE_OUTPUT_DIR = os.path.join(TestDataLocations.baseline_data_dir, "drill_in_box")

# Path where the generated models will be stored
TRAINING_OUTPUT_DIR = os.path.join(TestDataLocations.generated_data_dir, "training_output")
FRANKA_MESH_TRAINING_OUTPUT_DIR = os.path.join(TRAINING_OUTPUT_DIR, "mesh")
DRILL_IN_BOX_RGBD_TRAINING_OUTPUT_DIR = os.path.join(TRAINING_OUTPUT_DIR, "drill_in_box")


def test_datagen_franka_mesh(generate_baseline_arg):
    """Run replay to generate test data for the training tests below"""
    env = helpers.setup_datagen(FRANKA_DATASET_OUTPUT_DIR)
    helpers.run_datagen(
        output_dir=FRANKA_DATASET_OUTPUT_DIR,
        task=Tasks.CUBE_STACKING,
        data_type=DataType.MESH,
        add_external_cam=False,
        hdf5_filepath=TestDataLocations.cube_stacking_hdf5_filepath,
        env=env,
    )

    # Re-generate the baseline demo if requested. This should only be done when committing expected changes.
    if generate_baseline_arg:
        helpers.replace_baseline_dataset(
            generated_output_dir=FRANKA_DATASET_OUTPUT_DIR,
            baseline_output_dir=FRANKA_BASELINE_OUTPUT_DIR,
        )


def test_datagen_humanoid_2cam_rgbd(generate_baseline_arg):
    """Run replay to generate test data for the training tests below"""
    env = helpers.setup_datagen(DRILL_IN_BOX_DATASET_OUTPUT_DIR)
    helpers.run_datagen(
        output_dir=DRILL_IN_BOX_DATASET_OUTPUT_DIR,
        task=Tasks.DRILL_IN_BOX,
        data_type=DataType.RGBD,
        add_external_cam=True,
        hdf5_filepath=TestDataLocations.drill_in_box_hdf5_filepath,
        env=env,
    )

    # Re-generate the baseline demo if requested. This should only be done when committing expected changes.
    if generate_baseline_arg:
        helpers.replace_baseline_dataset(
            generated_output_dir=DRILL_IN_BOX_DATASET_OUTPUT_DIR,
            baseline_output_dir=DRILL_IN_BOX_BASELINE_OUTPUT_DIR,
        )


def test_dataset_equal_to_itself():
    """As a sanity check, make sure that the dataset is equal to itself"""
    arm_embodiment = ArmEmbodiment()
    is_equal = helpers.dataset_equal_to_baseline(
        arm_embodiment,
        task=Tasks.CUBE_STACKING,
        add_external_cam=False,
        demo_idx=DEMO_IDX,
        generated_dataset_dir=FRANKA_DATASET_OUTPUT_DIR,
        baseline_dataset_dir=FRANKA_DATASET_OUTPUT_DIR,
        data_type=DataType.MESH,
        batch_size=4,
        vertex_sampling_method=VertexSamplingMethod.RANDOM_WITHOUT_REPLACEMENT,
    )
    assert is_equal


@pytest.mark.skip(reason="Skipping this test until transforms are supported")
def test_dataset_not_equal_to_random_transformed():
    """Another sanity check: we make sure that dataset is not equal to itself if a random transform is applied"""
    arm_embodiment = ArmEmbodiment()
    is_equal = helpers.dataset_equal_to_baseline(
        arm_embodiment,
        task=Tasks.CUBE_STACKING,
        add_external_cam=False,
        demo_idx=DEMO_IDX,
        generated_dataset_dir=FRANKA_DATASET_OUTPUT_DIR,
        baseline_dataset_dir=FRANKA_DATASET_OUTPUT_DIR,
        data_type=DataType.MESH,
        batch_size=4,
        vertex_sampling_method=VertexSamplingMethod.RANDOM_WITHOUT_REPLACEMENT,
        apply_geometry_noise_to_baseline=True,
    )
    assert not is_equal


def test_dataset_equal_to_baseline_franka():
    """Compare the generated dataset with stored baseline. We expect them to be equal"""
    arm_embodiment = ArmEmbodiment()
    # We select all vertices (no sampling) to ensure datasets do not differ due to sampling.
    is_equal = helpers.dataset_equal_to_baseline(
        arm_embodiment,
        task=Tasks.CUBE_STACKING,
        add_external_cam=False,
        demo_idx=DEMO_IDX,
        generated_dataset_dir=FRANKA_DATASET_OUTPUT_DIR,
        baseline_dataset_dir=FRANKA_BASELINE_OUTPUT_DIR,
        data_type=DataType.MESH,
        batch_size=1,
        vertex_sampling_method=VertexSamplingMethod.NONE,
    )
    assert (
        is_equal
    ), f"ERROR: generated dataset differs from baseline. If this is expected, run \n pytest -s test_e2e.py -k test_datagen_franka --generate_baseline\n to replace the data."


def test_dataset_equal_to_baseline_humanoid():
    """Compare the generated dataset with stored baseline. We expect them to be equal"""
    humanoid_embodiment = HumanoidEmbodiment(Tasks.DRILL_IN_BOX)
    is_equal = helpers.dataset_equal_to_baseline(
        humanoid_embodiment,
        task=Tasks.DRILL_IN_BOX,
        add_external_cam=True,
        demo_idx=DEMO_IDX,
        generated_dataset_dir=DRILL_IN_BOX_DATASET_OUTPUT_DIR,
        baseline_dataset_dir=DRILL_IN_BOX_BASELINE_OUTPUT_DIR,
        data_type=DataType.RGBD,
        batch_size=1,
        vertex_sampling_method=VertexSamplingMethod.NONE,
    )
    assert (
        is_equal
    ), f"ERROR: generated dataset differs from baseline. If this is expected, run \n pytest -s test_e2e.py -k test_datagen_humanoid --generate_baseline\n to replace the data."


def test_training_franka_mesh():
    helpers.test_training(
        dataset=FRANKA_DATASET_OUTPUT_DIR,
        task=Tasks.CUBE_STACKING,
        data_type=DataType.MESH,
        add_external_cam=False,
        demo_idx=DEMO_IDX,
        training_output_dir=FRANKA_MESH_TRAINING_OUTPUT_DIR,
    )


def test_training_humanoid_2cam_rgbd():
    helpers.test_training(
        dataset=DRILL_IN_BOX_DATASET_OUTPUT_DIR,
        task=Tasks.DRILL_IN_BOX,
        data_type=DataType.RGBD,
        add_external_cam=False,
        demo_idx=DEMO_IDX,
        training_output_dir=DRILL_IN_BOX_RGBD_TRAINING_OUTPUT_DIR,
    )


def test_closed_loop_franka_mesh():
    checkpoint_filepath_glob = glob.glob(
        f"{FRANKA_MESH_TRAINING_OUTPUT_DIR}/checkpoints/*/last.pth"
    )
    assert (
        len(checkpoint_filepath_glob) == 1
    ), "No checkpoint found. has the training test been executed?"
    checkpoint_filepath = checkpoint_filepath_glob[0]
    print(f"Running closed loop test with checkpoint: {checkpoint_filepath}")
    helpers.test_closed_loop(
        checkpoint_filepath,
        task=Tasks.CUBE_STACKING,
        data_type=DataType.MESH,
        add_external_cam=False,
        hdf5_filepath=TestDataLocations.cube_stacking_hdf5_filepath,
    )


def test_closed_loop_humanoid_2cam_rgbd():
    checkpoint_filepath_glob = glob.glob(
        f"{DRILL_IN_BOX_RGBD_TRAINING_OUTPUT_DIR}/checkpoints/*/last.pth"
    )
    assert (
        len(checkpoint_filepath_glob) == 1
    ), "No checkpoint found. has the training test been executed?"
    checkpoint_filepath = checkpoint_filepath_glob[0]
    print(f"Running closed loop test with checkpoint: {checkpoint_filepath}")
    helpers.test_closed_loop(
        checkpoint_filepath,
        task=Tasks.DRILL_IN_BOX,
        data_type=DataType.RGBD,
        add_external_cam=True,
        hdf5_filepath=TestDataLocations.drill_in_box_hdf5_filepath,
    )


def test_dataloader_humanoid_2cam_rgbd():
    """ " Test history, batch size and prediction horizon works"""
    batch_size = 3
    num_history = 2
    prediction_horizon = 1  # keypose mode only predicts one timestep
    dataloader = helpers.get_dataloader(
        embodiment=HumanoidEmbodiment(Tasks.DRILL_IN_BOX),
        task=Tasks.DRILL_IN_BOX,
        data_type=DataType.RGBD,
        add_external_cam=True,
        dataset_path=DRILL_IN_BOX_DATASET_OUTPUT_DIR,
        demos=str(DEMO_IDX),
        batch_size=batch_size,
        num_history=num_history,
        prediction_horizon=prediction_horizon,
    )

    for batch in dataloader:
        helpers.validate_humanoid_batch(
            batch, batch_size, num_history, prediction_horizon, add_external_cam=True
        )


def test_dataloader_humanoid_1cam_rgbd():
    """ " Test history, batch size and prediction horizon works"""
    batch_size = 3
    num_history = 2
    prediction_horizon = 1  # keypose mode only predicts one timestep
    dataloader = helpers.get_dataloader(
        embodiment=HumanoidEmbodiment(Tasks.DRILL_IN_BOX),
        task=Tasks.DRILL_IN_BOX,
        data_type=DataType.RGBD,
        add_external_cam=False,
        dataset_path=DRILL_IN_BOX_DATASET_OUTPUT_DIR,
        demos=str(DEMO_IDX),
        batch_size=batch_size,
        num_history=num_history,
        prediction_horizon=prediction_horizon,
    )

    for batch in dataloader:
        helpers.validate_humanoid_batch(
            batch, batch_size, num_history, prediction_horizon, add_external_cam=False
        )


def test_dataloader_humanoid_1cam_rgbd_only_keypose():
    """ "Check that only-keypose mode works"""

    # First let's run without "only_keyposes" and collect keypose batches
    dataloader = helpers.get_dataloader(
        embodiment=HumanoidEmbodiment(Tasks.DRILL_IN_BOX),
        task=Tasks.DRILL_IN_BOX,
        data_type=DataType.RGBD,
        add_external_cam=False,
        dataset_path=DRILL_IN_BOX_DATASET_OUTPUT_DIR,
        demos=str(DEMO_IDX),
        only_sample_keyposes=False,
        use_keyposes=True,
        batch_size=1,
    )
    batches = []
    for batch in dataloader:
        if batch[IS_KEYPOSE_ITEM_NAME][0] == True:
            batches.append(batch)

    # Now run with only_keyposes and collect batches
    dataloader_only_keyposes = helpers.get_dataloader(
        embodiment=HumanoidEmbodiment(Tasks.DRILL_IN_BOX),
        task=Tasks.DRILL_IN_BOX,
        data_type=DataType.RGBD,
        add_external_cam=False,
        dataset_path=DRILL_IN_BOX_DATASET_OUTPUT_DIR,
        demos=str(DEMO_IDX),
        only_sample_keyposes=True,
        use_keyposes=True,
        batch_size=1,
    )
    keypose_only_batches = []
    for batch in dataloader_only_keyposes:
        keypose_only_batches.append(batch)

    # All batches should be identical
    for s1, s2 in zip(batches, keypose_only_batches):
        for key in s1.keys():
            assert torch.all(s1[key] == s2[key])
        for key in s2.keys():
            assert torch.all(s1[key] == s2[key])


def test_validate_demos_franka_mesh():
    helpers.test_validate_demos(
        dataset_path=FRANKA_DATASET_OUTPUT_DIR,
        task=Tasks.CUBE_STACKING,
        data_type=DataType.MESH,
    )

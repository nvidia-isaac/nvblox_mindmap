# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import os

import pytest
import torch

from mindmap.data_loading.data_types import DataType
from mindmap.data_loading.dataset import SamplingWeightingType
from mindmap.data_loading.dataset_files_by_encoding_method import get_data_loader_by_data_type
from mindmap.data_loading.item_names import (
    GT_POLICY_STATE_PRED_ITEM_NAME,
    NVBLOX_VERTEX_FEATURES_ITEM_NAME,
    POLICY_STATE_HISTORY_ITEM_NAME,
)
from mindmap.data_loading.vertex_sampling import VertexSamplingMethod
from mindmap.embodiments.arm.embodiment import ArmEmbodiment
from mindmap.geometry.pytorch3d_transforms import (
    matrix_to_euler_angles,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_matrix,
)
from mindmap.keyposes.keypose_detection_mode import KeyposeDetectionMode
from mindmap.tasks.tasks import Tasks
from mindmap.tests.utils.constants import TestDataLocations

DEMO_DIR = os.path.join(f"{TestDataLocations.generated_data_dir}/franka/", "demo_00000")


def _get_dataloader(
    data_type,
    batch_size,
    num_history,
    prediction_horizon,
    apply_random_transforms,
    apply_geometry_noise,
    random_translation_range_m=None,
    random_rpy_range_deg=None,
    pos_noise_stddev_m=0,
    rot_noise_stddev_deg=0,
):
    """Get dataloaders used during testing"""
    train_loader, _ = get_data_loader_by_data_type(
        embodiment=ArmEmbodiment(),  # Test with arm embodiment
        task=Tasks.CUBE_STACKING,
        dataset_path=DEMO_DIR,
        num_workers=0,
        batch_size=batch_size,
        use_keyposes=False,
        data_type=data_type,
        only_sample_keyposes=False,
        extra_keyposes_around_grasp_events=[],
        keypose_detection_mode=KeyposeDetectionMode.HIGHEST_Z_BETWEEN_GRASP,
        include_failed_demos=False,
        sampling_weighting_type=SamplingWeightingType.NONE,
        gripper_encoding_mode="binary",
        num_history=num_history,
        prediction_horizon=prediction_horizon,
        apply_random_transforms=apply_random_transforms,
        apply_geometry_noise=apply_geometry_noise,
        pos_noise_stddev_m=pos_noise_stddev_m,
        rot_noise_stddev_deg=rot_noise_stddev_deg,
        add_external_cam=True,
        num_vertices_to_sample=1024,
        vertex_sampling_method=VertexSamplingMethod.RANDOM_WITHOUT_REPLACEMENT,
        random_translation_range_m=random_translation_range_m,
        random_rpy_range_deg=random_rpy_range_deg,
        seed=0,
    )
    return train_loader


def check_pcd_transform(
    original_batch,
    transformed_batch,
    sample_type_idx,
    sample_idx,
    random_translation,
    random_rotation,
    permute=False,
):
    """Check that a point cloud was correctly transformed by the random transform."""
    # NOTE: pcd: T_AW, random_transformed_pcd: T_BW, random_transform: T_BA
    pcd = original_batch[sample_type_idx][sample_idx]
    random_transformed_pcd = transformed_batch[sample_type_idx][sample_idx]

    # For the wrist/table PCD, we need to permute the dimensions.
    if permute:
        pcd = pcd.permute(1, 2, 0)
        random_transformed_pcd = random_transformed_pcd.permute(1, 2, 0)

    # Compute the transformed point cloud.
    # B_t_BW = R_BA * A_t_AW + B_t_BA
    computed_transformed_pcd = quaternion_apply(random_rotation, pcd) + random_translation

    # Check that the transformed point cloud matches the expected result.
    assert torch.allclose(computed_transformed_pcd, random_transformed_pcd, atol=1e-3)


def check_vertices_transform(
    original_batch,
    transformed_batch,
    sample_type_idx,
    sample_idx,
    random_translation,
    random_rotation,
    permute=False,
):
    """Check that a point cloud was correctly transformed by the random transform."""
    # NOTE: pcd: T_AW, random_transformed_pcd: T_BW, random_transform: T_BA
    pcd = original_batch[sample_type_idx]["vertices"][sample_idx]
    random_transformed_pcd = transformed_batch[sample_type_idx]["vertices"][sample_idx]

    # For the wrist/table PCD, we need to permute the dimensions.
    if permute:
        pcd = pcd.permute(1, 2, 0)
        random_transformed_pcd = random_transformed_pcd.permute(1, 2, 0)

    # Compute the transformed point cloud.
    # B_t_BW = R_BA * A_t_AW + B_t_BA
    computed_transformed_pcd = quaternion_apply(random_rotation, pcd) + random_translation

    # Check that the transformed point cloud matches the expected result.
    assert torch.allclose(
        computed_transformed_pcd.to(dtype=random_transformed_pcd.dtype),
        random_transformed_pcd,
        atol=1e-3,
    )


def check_trajectory_transform(
    original_batch,
    transformed_batch,
    sample_type_idx,
    sample_idx,
    timestep_idx,
    random_translation,
    random_rotation,
):
    """Check that a trajectory was correctly transformed by the random transform."""
    # NOTE: initial_pose: T_AW, transformed_pose: T_BW, random_transform: T_BA
    initial_pose = original_batch[sample_type_idx][sample_idx][timestep_idx]
    transformed_pose = transformed_batch[sample_type_idx][sample_idx][timestep_idx]

    # Compute the translation and rotation that transforms the initial pose into the transformed pose.
    trans, rot = compute_transform(initial_pose, transformed_pose)

    # Check that the gripper state is unchanged.
    assert initial_pose[7] == transformed_pose[7]

    # Check that the translation and rotation are correct.
    assert torch.allclose(trans, random_translation, atol=1e-3)
    assert torch.allclose(rot, random_rotation, atol=1e-3) or torch.allclose(
        rot, -random_rotation, atol=1e-3
    )


def compute_transform(initial_pose, transformed_pose):
    """Compute the translation and rotation that transforms the initial pose into the transformed pose."""
    # NOTE: initial_pose: T_AW, transformed_pose: T_BW, transform to compute T_BA
    # R_BA = R_BW * inv(R_AW)
    rotation = quaternion_multiply(transformed_pose[3:7], quaternion_invert(initial_pose[3:7]))
    # B_t_BA = B_t_BW - R_BA * A_t_AW
    translation = transformed_pose[:3] - quaternion_apply(rotation, initial_pose[:3])
    return translation, rotation


def _test_random_transform_augmentation(
    batch_size,
    num_history,
    prediction_horizon,
    random_translation_range_m,
    random_rpy_range_deg,
    data_type,
):
    """
    Test that the same random transform is applied to all elements of one sample
    and that the random transform is valid
    """

    # Get one dataloader with no random transforms and one with random transforms.
    dataloader = _get_dataloader(
        data_type=data_type,
        batch_size=batch_size,
        num_history=num_history,
        prediction_horizon=prediction_horizon,
        apply_random_transforms=False,
        random_translation_range_m=random_translation_range_m,
        random_rpy_range_deg=random_rpy_range_deg,
        apply_geometry_noise=False,
    )
    random_dataloader = _get_dataloader(
        data_type=data_type,
        batch_size=batch_size,
        num_history=num_history,
        prediction_horizon=prediction_horizon,
        apply_random_transforms=True,
        random_translation_range_m=random_translation_range_m,
        random_rpy_range_deg=random_rpy_range_deg,
        apply_geometry_noise=False,
    )
    assert len(dataloader) == len(random_dataloader)

    for batch, random_batch in zip(dataloader, random_dataloader):
        for sample_idx in range(batch_size):
            # Compute the random transform applied to this sample (i.e. T_BA).
            random_translation, random_rotation = compute_transform(
                initial_pose=batch[POLICY_STATE_HISTORY_ITEM_NAME][sample_idx][0],
                transformed_pose=random_batch[POLICY_STATE_HISTORY_ITEM_NAME][sample_idx][0],
            )

            # Check that the translation is within the range
            assert torch.all(
                random_translation >= torch.tensor(random_translation_range_m[0]) - 1e-6
            )
            assert torch.all(
                random_translation <= torch.tensor(random_translation_range_m[1]) + 1e-6
            )
            assert not torch.all(random_translation == 0)

            # Check that the rotation is within the range
            rotation_matrix = quaternion_to_matrix(random_rotation)
            random_rotation_rpy = matrix_to_euler_angles(rotation_matrix, "XYZ")
            assert torch.allclose(torch.norm(random_rotation), torch.tensor(1.0), atol=1e-3)
            assert torch.all(random_rotation_rpy >= torch.tensor(random_rpy_range_deg[0]) - 1e-3)
            assert torch.all(random_rotation_rpy <= torch.tensor(random_rpy_range_deg[1]) + 1e-3)
            assert not torch.all(random_rotation_rpy == 0)

            # Check that the applied transform is consistent for the gripper history.
            for hist_idx in range(num_history):
                check_trajectory_transform(
                    batch,
                    random_batch,
                    POLICY_STATE_HISTORY_ITEM_NAME,
                    sample_idx,
                    hist_idx,
                    random_translation,
                    random_rotation,
                )

            # Check that the applied transform is consistent for the GT trajectory.
            for pred_idx in range(prediction_horizon):
                check_trajectory_transform(
                    batch,
                    random_batch,
                    GT_POLICY_STATE_PRED_ITEM_NAME,
                    sample_idx,
                    pred_idx,
                    random_translation,
                    random_rotation,
                )

            if data_type == DataType.MESH:
                # Check that the applied transform is consistent for the mesh vertices.
                check_vertices_transform(
                    batch,
                    random_batch,
                    NVBLOX_VERTEX_FEATURES_ITEM_NAME,
                    sample_idx,
                    random_translation,
                    random_rotation,
                )


def _test_diff_is_gaussian(samples1, samples2, expected_stddev):
    stddev_compute = torch.std(samples1 - samples2)
    mean_compute = torch.mean(samples1 - samples2)
    assert abs(stddev_compute - expected_stddev) < 1e-3
    assert abs(mean_compute) < 1e-3


def _test_noising(
    batch_size,
    num_history,
    prediction_horizon,
    pos_noise_stddev_m,
    rot_noise_stddev_deg,
    data_type,
):
    """
    Test that the same random transform is applied to all elements of one sample
    and that the random transform is valid
    """

    # Get one dataloader with no random transforms and one with random transforms.
    dataloader = _get_dataloader(
        data_type=data_type,
        batch_size=batch_size,
        num_history=num_history,
        prediction_horizon=prediction_horizon,
        apply_geometry_noise=False,
        apply_random_transforms=False,
        pos_noise_stddev_m=pos_noise_stddev_m,
        rot_noise_stddev_deg=rot_noise_stddev_deg,
    )
    noise_dataloader = _get_dataloader(
        data_type=data_type,
        batch_size=batch_size,
        num_history=num_history,
        prediction_horizon=prediction_horizon,
        apply_geometry_noise=True,
        apply_random_transforms=False,
        pos_noise_stddev_m=pos_noise_stddev_m,
        rot_noise_stddev_deg=rot_noise_stddev_deg,
    )

    assert len(dataloader) == len(noise_dataloader)

    gripper_pos = []
    gripper_pos_noise = []
    for batch, noise_batch in zip(dataloader, noise_dataloader):
        for sample_idx in range(batch_size):
            # Test vertices
            _test_diff_is_gaussian(
                batch[NVBLOX_VERTEX_FEATURES_ITEM_NAME]["vertices"].view(-1, 1),
                noise_batch[NVBLOX_VERTEX_FEATURES_ITEM_NAME]["vertices"].view(-1, 1),
                pos_noise_stddev_m,
            )

            # Test poses
            gripper_pos.extend(
                [t.item() for t in batch[POLICY_STATE_HISTORY_ITEM_NAME][:, :, :3].reshape(-1, 1)]
            )
            gripper_pos_noise.extend(
                [
                    t.item()
                    for t in noise_batch[POLICY_STATE_HISTORY_ITEM_NAME][:, :, :3].reshape(-1, 1)
                ]
            )

    _test_diff_is_gaussian(
        torch.tensor(gripper_pos), torch.tensor(gripper_pos_noise), pos_noise_stddev_m
    )


@pytest.mark.skip(reason="Skipping this test until transforms are supported")
def test_random_transform_augmentation():
    """
    Test that the same random transform is applied to all elements of one sample
    and that the random transform is valid for voxel grid and PCD modes.
    """
    batch_size = 3
    num_history = 2
    prediction_horizon = 4
    random_translation_range_m = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    random_rpy_range_deg = [[-180.0, -180.0, -180.0], [180.0, 180.0, 180.0]]
    _test_random_transform_augmentation(
        batch_size,
        num_history,
        prediction_horizon,
        random_translation_range_m,
        random_rpy_range_deg,
        data_type=DataType.MESH,
    )


@pytest.mark.skip(reason="Skipping this test until transforms are supported")
def test_noising():
    batch_size = 3
    num_history = 2
    prediction_horizon = 4
    pos_stddev_m = 1e-3
    rot_stddev_deg = 0

    # Check that the poses are transformed correctly for mesh vertices mode.
    print("Testing noising augmentation for mesh vertices mode...")
    # Only apply 2d transform this time.
    _test_noising(
        batch_size,
        num_history,
        prediction_horizon,
        pos_stddev_m,
        rot_stddev_deg,
        data_type=DataType.MESH,
    )

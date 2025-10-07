# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from abc import ABC, abstractmethod
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F

from mindmap.data_loading.vertex_sampling import VertexSamplingMethod, sample_to_n_vertices
from mindmap.geometry.pytorch3d_transforms import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_multiply,
)
from mindmap.image_processing.image_conversions import convert_rgb_to_model_input
from mindmap.mapping.nvblox_mapper_constants import DEPTH_SCALE_FACTOR


class SampleTransformer(ABC):
    """Base class for sample transformers"""

    def reset(self):
        """Implement this function if the transformer contains a state that should be reset before
        transforming a sample"""
        pass

    @abstractmethod
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Override this function to transform samples"""
        pass


class RgbTransformer(SampleTransformer):
    """Transform an image"""

    def __call__(self, image):
        """
        Transforms an image tensor:
        - converts range [0, 255] (IsaacLab) to [0, 1]
        - moves channel dimension to front (from [H, W, C] to [C, H, W])

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        return convert_rgb_to_model_input(image)


class DepthTransformer(SampleTransformer):
    """Transform a depth image"""

    def __call__(self, image):
        """
        Transforms a depth image tensor by scaling it from uint16 values to floating point meters.

        Args:
            image (torch.Tensor): The input depth image tensor in uint16 format, scaled by DEPTH_SCALE_FACTOR.

        Returns:
            torch.Tensor: The transformed depth image tensor in float32 format, representing depths in meters.
        """
        return (image / DEPTH_SCALE_FACTOR).to(torch.float32)


class GeometryAugmentor(SampleTransformer):
    """Augment geometry input by transforming everything with the same transform, drawn from a
    uniform distribution. Call reset() to re-compute the random transform"""

    def __init__(
        self,
        random_translation_range_m: Tuple[List[float], List[float]],
        random_rpy_range_deg: Tuple[List[float], List[float]],
    ):
        """
        Args:
           random_translation_range_m: Bounds of random translation
           random_rpy_range_deg: Bounds of random rotation"""

        self._random_translation_range_m = random_translation_range_m
        self._random_rpy_range_deg = random_rpy_range_deg
        self._random_transform = None
        self.reset()

    def reset(self):
        """Recompute the transform. Should be called once before a new sample is processed"""
        if self._random_rpy_range_deg is not None and self._random_rpy_range_deg is not None:
            self._random_transform = random_transform_uniform(
                self._random_translation_range_m, self._random_rpy_range_deg
            )

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply the transform to a sample"""
        # Handle both mesh vertices and poses
        sample_tensor = sample["vertices"] if isinstance(sample, dict) else sample
        sample_tensor = apply_random_transform_to_sample(
            sample_tensor, self._random_transform[0], self._random_transform[1]
        )
        if isinstance(sample, dict):
            sample["vertices"] = sample_tensor
        else:
            sample = sample_tensor

        return sample


class GeometryNoiser(SampleTransformer):
    """Add noise to geometry (poses and 3d-positions) by drawing independent samples from a Gaussian
    distribution."""

    def __init__(self, pos_stddev_m, rot_stddev_deg):
        """
        Args:
            pos_stddev_m:  Standard dev of position noise
            rot_stddev_deg: Standard dev of rotation noise
        """
        self._pos_stddev_m = pos_stddev_m
        self._rot_stddev_deg = rot_stddev_deg

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply the transform to a sample"""
        # Handle both mesh vertices and poses
        sample_tensor = sample["vertices"] if isinstance(sample, dict) else sample

        # Create N transforms - one for each object we're transforming.
        random_transforms = random_transform_gaussian(
            self._pos_stddev_m, self._rot_stddev_deg, sample_tensor.shape[0]
        )
        sample_tensor = apply_random_transform_to_sample(
            sample_tensor, random_transforms[0], random_transforms[1]
        )
        if isinstance(sample, dict):
            sample["vertices"] = sample_tensor
        else:
            sample = sample_tensor

        return sample


class VertexSampler(SampleTransformer):
    """Reduces or increases the number of vertices in a sample to a fixed number."""

    def __init__(self, desired_num_vertices: int, method: VertexSamplingMethod, seed: int = None):
        """
        Args:
            desired_num_vertices: Number of vertices the sample should contain after subsampling.
            method: The sampling method.
        """
        assert isinstance(
            method, VertexSamplingMethod
        ), "Require vertex_sampling_method when using mesh."
        if method != VertexSamplingMethod.NONE:
            assert (
                desired_num_vertices is not None
            ), "Require num_vertices_to_sample when using mesh."
            assert desired_num_vertices > 0, "Require num_vertices_to_sample to be greater than 0."
        self.desired_num_vertices = desired_num_vertices
        self.method = method
        self.seed = seed

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        (
            sample["vertices"],
            sample["features"],
            sample["vertices_valid_mask"],
        ) = sample_to_n_vertices(
            sample["vertices"],
            sample["features"],
            self.desired_num_vertices,
            self.method,
            self.seed,
        )
        if self.method != VertexSamplingMethod.NONE:
            assert sample["vertices"].shape[0] == self.desired_num_vertices
            assert sample["features"].shape[0] == self.desired_num_vertices
        return sample


def random_transform_uniform(
    random_translation_range_m: Tuple[List[float], List[float]],
    random_rpy_range_deg: Tuple[List[float], List[float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random translation and quaternion.

    Args:
        random_translation_range_m (Tuple[List[float], List[float]]): The range of the random translation.
        random_rpy_range_deg (Tuple[List[float], List[float]]): The range of the random rotation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the translation vector and rotation quaternion.
    """
    # Random translation
    translation = torch.tensor(
        [
            random.uniform(random_translation_range_m[0][i], random_translation_range_m[1][i])
            for i in range(3)
        ]
    )

    # Random rotation
    rotation_degrees = torch.tensor(
        [random.uniform(random_rpy_range_deg[0][i], random_rpy_range_deg[1][i]) for i in range(3)]
    )
    rotation_radians = torch.deg2rad(rotation_degrees)
    rotation_matrix = euler_angles_to_matrix(rotation_radians, "XYZ")
    quaternion = matrix_to_quaternion(rotation_matrix)

    return translation, quaternion


def random_transform_gaussian(
    pos_stddev_m: float, rot_stddev_deg: float, num_transforms: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random gausian form a Gaussian distribution with mean zero and given standard deviation

    Args:
        pos_stddev_m: Standard deviation of position
        rot_stddev_deg: Standard deviation of rotation

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the translation vector and rotation quaternion.
    """
    # Random translation
    shape = (num_transforms, 3)
    translation = torch.normal(mean=torch.zeros(shape), std=torch.full(shape, pos_stddev_m))

    # Random rotation
    rot_stddev_rad = torch.deg2rad(torch.tensor(rot_stddev_deg))
    rotation_rad = torch.normal(mean=torch.zeros(shape), std=torch.full(shape, rot_stddev_rad))
    rotation_matrix = euler_angles_to_matrix(rotation_rad, "XYZ")
    quaternion = matrix_to_quaternion(rotation_matrix)

    return translation, quaternion


def apply_random_transform_to_sample(
    sample: torch.Tensor, random_translation: torch.Tensor, random_rotation: torch.Tensor
) -> torch.Tensor:
    """
    Apply random translation and rotation to the sample.

    Args:
        sample (torch.Tensor): The sample to transform.
        random_translation (torch.Tensor): The random translation vector.
        random_rotation (torch.Tensor): The random rotation quaternion.

    Returns:
        torch.Tensor: The transformed sample.
    """
    # NOTE: initial_pose: T_AW, transformed_pose: T_BW, random_transform: T_BA
    # Either translation only or translation + quaternion + gripper state
    assert sample.shape[-1] in [3, 8]

    original_dtype = sample.dtype

    # Apply rotation to the translation part
    # B_t_BW = R_BA * A_t_AW + B_t_BA
    translation = sample[..., :3]
    transformed_translation = quaternion_apply(random_rotation, translation) + random_translation

    if sample.shape[-1] == 8:
        quaternion = sample[..., 3:7]
        gripper_state = sample[..., 7:]

        # Apply rotation to the quaternion part
        # R_BW = R_BA * R_AW
        rotated_quaternion = quaternion_multiply(random_rotation, quaternion)

        # Concatenate the transformed parts with the unchanged gripper state
        transformed_sample = torch.cat(
            [transformed_translation, rotated_quaternion, gripper_state], dim=-1
        )
    else:
        transformed_sample = transformed_translation

    assert transformed_sample.shape == sample.shape

    return transformed_sample.to(dtype=original_dtype)

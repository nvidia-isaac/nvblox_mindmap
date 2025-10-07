# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import Tuple

import torch

from mindmap.data_loading.vertex_sampling import VertexSamplingMethod, sample_to_n_vertices


def _get_test_vertices(num_vertices: int) -> Tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.ones(num_vertices, 3)
    features = torch.ones(num_vertices, 120)
    for i in range(num_vertices):
        vertices[i, :] *= i + 1
        features[i, :] *= i + 1
    return vertices, features


def _are_rows_unique(tensor: torch.Tensor) -> bool:
    return torch.unique(tensor, dim=0).shape[0] == tensor.shape[0]


def _count_zero_rows(tensor: torch.Tensor) -> int:
    num_rows = tensor.shape[0]
    num_zero_rows = 0
    for i in range(num_rows):
        if torch.all(tensor[i, :] == 0):
            num_zero_rows += 1
    return num_zero_rows


def test_random_sampling_without_replacement():
    # Test vectors
    vertices, features = _get_test_vertices(1000)
    # Sample
    num_desired_vertices = 500
    vertices, features, _ = sample_to_n_vertices(
        vertices,
        features,
        num_desired_vertices,
        VertexSamplingMethod.RANDOM_WITHOUT_REPLACEMENT,
    )
    # Check sizes
    assert vertices.shape[0] == num_desired_vertices
    assert vertices.shape[1] == 3
    assert features.shape[0] == num_desired_vertices
    assert features.shape[1] == 120
    # Check uniqueness
    assert _are_rows_unique(vertices)
    assert _are_rows_unique(features)


def test_random_sampling_with_replacement():
    # Test vectors
    vertices, features = _get_test_vertices(1000)
    # Sample
    num_desired_vertices = 500
    vertices, features, _ = sample_to_n_vertices(
        vertices,
        features,
        num_desired_vertices,
        VertexSamplingMethod.RANDOM_WITH_REPLACEMENT,
    )
    # Check sizes
    assert vertices.shape[0] == num_desired_vertices
    assert vertices.shape[1] == 3
    assert features.shape[0] == num_desired_vertices
    assert features.shape[1] == 120


def test_padding():
    # Test vectors
    num_vertices = 100
    vertices, features = _get_test_vertices(num_vertices)
    # Sample
    num_desired_vertices = 500
    vertices, features, valid_mask = sample_to_n_vertices(
        vertices,
        features,
        num_desired_vertices,
        VertexSamplingMethod.RANDOM_WITHOUT_REPLACEMENT,
    )
    # Check sizes
    assert vertices.shape[0] == num_desired_vertices
    assert vertices.shape[1] == 3
    assert features.shape[0] == num_desired_vertices
    assert features.shape[1] == 120
    # Check padded rows
    assert _count_zero_rows(vertices) == (num_desired_vertices - num_vertices)
    assert _count_zero_rows(features) == (num_desired_vertices - num_vertices)

    # Check mask
    assert valid_mask.shape[0] == num_desired_vertices

    for tensor in [vertices, features]:
        assert torch.all(tensor[~valid_mask] == 0)
        assert not torch.any(tensor[valid_mask] == 0)


def test_do_nothing():
    # Test vectors
    num_vertices = 100
    vertices, features = _get_test_vertices(num_vertices)
    # Sample
    num_desired_vertices = 10
    vertices_sampled, features_sampled, _ = sample_to_n_vertices(
        vertices, features, num_desired_vertices, VertexSamplingMethod.NONE
    )
    # Check that nothing has happened
    assert vertices_sampled.shape[0] == num_vertices
    assert vertices_sampled.shape[1] == 3
    assert features_sampled.shape[0] == num_vertices
    assert features_sampled.shape[1] == 120
    assert torch.all(vertices_sampled == vertices)
    assert torch.all(features_sampled == features)

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from enum import Enum
from typing import Tuple

import numpy as np
import torch


class VertexSamplingMethod(Enum):
    """Method to encode the features for the Diffuser Actor network."""

    # TODO(alexmillane, 2025-03-26): Once experiments confirm that `random_with_replacement` is
    # doesn't have some surprising benefit, we should remove it and change
    # `random_with_replacement` to `random`.
    RANDOM_WITHOUT_REPLACEMENT = "random_without_replacement"
    RANDOM_WITH_REPLACEMENT = "random_with_replacement"
    LOWEST = "lowest"
    NONE = "none"


def sample_to_n_vertices(
    vertices: torch.Tensor,
    features: torch.Tensor,
    desired_num_vertices: int,
    method: VertexSamplingMethod,
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Samples N (vertices + features) to the requested number. If the number of vertices in the input is less than
    the requested amount, zero rows are added to make the number of rows match.

    Args:
        vertices (torch.Tensor): An Nx3 tensor of vertex positions.
        features (torch.Tensor): An NxC tensor of vertex features.
        desired_num_vertices (int): The desired number of vertices.
        method (VertexSamplingMethod): The method used for the sampling.
        seed (int): The seed for the random sampling.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Mx3, MxC, and Mx1 tensors where M is the requested vertex number.
            Note is the requested methof is VertexSamplingMethod.NONE, the input vectors are returned.
    """
    assert vertices.dim() == 2
    assert features.dim() == 2
    assert vertices.shape[0] == features.shape[0]
    num_features = features.shape[0]
    if method == VertexSamplingMethod.NONE or num_features == desired_num_vertices:
        # If no sampling requested, Return input data and a
        valid_mask = torch.ones(num_features, device=vertices.device, dtype=torch.bool)
        return vertices, features, valid_mask
    elif num_features > desired_num_vertices:
        # If we are sampling down, run the appropriate sampling method and c
        valid_mask = torch.ones(desired_num_vertices, device=vertices.device, dtype=torch.bool)
        if method == VertexSamplingMethod.RANDOM_WITHOUT_REPLACEMENT:
            vertices, features = select_n_random_without_replacement(
                vertices, features, desired_num_vertices, seed
            )
        elif method == VertexSamplingMethod.RANDOM_WITH_REPLACEMENT:
            vertices, features = select_n_random_with_replacement(
                vertices, features, desired_num_vertices, seed
            )
        elif method == VertexSamplingMethod.LOWEST:
            vertices, features = select_n_lowest_z_vertices(
                vertices, features, desired_num_vertices
            )
        else:
            raise ValueError(f"Vertex sampling method {method} is not yet implemented.")
    else:
        # Pad with zeros if we don't have enough vertices
        vertices, features, valid_mask = pad_with_zeros(vertices, features, desired_num_vertices)

    assert vertices.shape[0] == desired_num_vertices
    assert features.shape[0] == desired_num_vertices

    return vertices, features, valid_mask


def pad_with_zeros(
    vertices: torch.Tensor, features: torch.Tensor, desired_num_vertices: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select the vertices and features with zero rows to bring the number to desired_num_vertices.
    """
    assert vertices.dim() == 2
    assert features.dim() == 2
    assert vertices.shape[0] == vertices.shape[0]
    num_features = features.shape[0]
    assert num_features < desired_num_vertices
    num_samples_to_add = desired_num_vertices - num_features
    zero_features = torch.zeros((num_samples_to_add, features.shape[1]), device=features.device)
    features = torch.cat([features, zero_features], dim=0)
    zero_vertices = torch.zeros((num_samples_to_add, vertices.shape[1]), device=vertices.device)
    vertices = torch.cat([vertices, zero_vertices], dim=0)

    # Create mask to indicate padded vertices
    valid_mask = torch.ones(vertices.shape[0], device=vertices.device, dtype=torch.bool).squeeze(-1)
    valid_mask[num_features:] = False

    assert torch.all(vertices[~valid_mask] == 0)
    assert torch.all(features[~valid_mask, :] == 0)
    return vertices, features, valid_mask


def select_n_lowest_z_vertices(
    vertices: torch.Tensor, features: torch.Tensor, number_of_vertices_to_select: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select the vertices with the lowest z-coordinate from the given vertices and features.
    """
    assert vertices.dim() == 2
    assert features.dim() == 2
    assert vertices.shape[0] == vertices.shape[0]
    assert vertices.shape[0] >= number_of_vertices_to_select
    assert vertices.shape[1] == 3
    selected_vertice_indices = np.argsort(-vertices[:, 2])[:number_of_vertices_to_select]
    vertices = vertices[selected_vertice_indices, :]
    features = features[selected_vertice_indices, :]

    return vertices, features


def select_n_random_without_replacement(
    vertices: torch.Tensor,
    features: torch.Tensor,
    number_of_vertices_to_select: int,
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select a number of random vertices and features without replacement.
    """
    assert vertices.dim() == 2
    assert features.dim() == 2
    assert vertices.shape[0] == vertices.shape[0]
    num_vertices = vertices.shape[0]
    assert num_vertices >= number_of_vertices_to_select
    if seed is not None:
        torch.manual_seed(seed)
    sampled_indices = torch.randperm(num_vertices)[:number_of_vertices_to_select]
    features = features[sampled_indices, :]
    vertices = vertices[sampled_indices, :]
    return vertices, features


def select_n_random_with_replacement(
    vertices: torch.Tensor,
    features: torch.Tensor,
    number_of_vertices_to_select: int,
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select a number of random vertices and features without replacement.
    """
    assert vertices.dim() == 2
    assert features.dim() == 2
    assert vertices.shape[0] == vertices.shape[0]
    num_vertices = vertices.shape[0]
    assert num_vertices >= number_of_vertices_to_select
    if seed is not None:
        torch.manual_seed(seed)
    sampled_indices = torch.randint(0, num_vertices, (number_of_vertices_to_select,))
    features = features[sampled_indices, :]
    vertices = vertices[sampled_indices, :]
    return vertices, features

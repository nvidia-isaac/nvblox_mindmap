# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import math
from typing import Tuple

from nvblox_torch.layer import FeatureLayer, Layer, convert_layer_to_dense_tensor
from nvblox_torch.mapper import Mapper
import torch

from mindmap.data_loading.vertex_sampling import VertexSamplingMethod, sample_to_n_vertices
from mindmap.mapping.helpers.nvblox_mapping_helpers import MAPPER_TO_ID
from mindmap.mapping.nvblox_mapper_constants import NvbloxMappingCfg


def get_vertices_and_features(
    mapper: Mapper,
    mapper_id: int,
    nvblox_mapping_config: NvbloxMappingCfg,
    remove_zero_features: bool,
    num_excess_features: int,
    sample_vertices: bool,
    number_of_vertices_to_sample: int = None,
    vertex_sampling_method: VertexSamplingMethod = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract vertices and features from the nvblox mapper.

    This function gets the mesh vertices from the mapper, filters them based on the AABB bounds,
    and returns the vertices along with their corresponding features.

    Args:
        mapper (Mapper): The nvblox mapper instance
        mapper_id (int): ID of the mapper to query (static vs dynamic)
        nvblox_mapping_config (NvbloxMappingCfg): Configuration containing mapping parameters,

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - vertices: Filtered mesh vertices tensor
            - features: Corresponding features tensor for the vertices
    """
    # Get the mesh vertices and features
    mapper.update_feature_mesh(mapper_id)
    mesh = mapper.get_feature_mesh(mapper_id)
    vertices = mesh.vertices()
    features = mesh.vertex_features()
    assert vertices.shape[0] == features.shape[0]
    assert vertices.shape[0] != 0, f"No vertices found in the mesh."
    assert vertices.is_cuda and features.is_cuda

    # Filter vertices and features based on AABB bounds
    aabb_min_m = nvblox_mapping_config.aabb_min_m.to("cuda")
    aabb_max_m = nvblox_mapping_config.aabb_max_m.to("cuda")
    mask = torch.all(torch.logical_and(vertices > aabb_min_m, vertices < aabb_max_m), dim=1)
    vertices = vertices[mask]
    features = features[mask]

    # Remove excess features if applicable
    if num_excess_features > 0:
        features = features[..., :-num_excess_features]

    # Remove features with all zero values
    if remove_zero_features:
        # Find indices where all feature values are zero
        zero_feature_mask = torch.all(features == 0, dim=1)
        # Remove vertices and features with zero features
        vertices = vertices[~zero_feature_mask]
        features = features[~zero_feature_mask]

    if not sample_vertices:
        # Valid everywhere
        valid_mask = torch.ones(
            vertices.shape[0], dtype=torch.bool, device=vertices.device
        ).unsqueeze(0)
    else:
        vertices, features, valid_mask = sample_to_n_vertices(
            vertices, features, number_of_vertices_to_sample, vertex_sampling_method
        )
        # Add batch dimension if needed (e.g. if we're running single batch)
        if valid_mask.ndim == 1:
            vertices = vertices.unsqueeze(0)
            features = features.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0)

    return vertices, features, valid_mask

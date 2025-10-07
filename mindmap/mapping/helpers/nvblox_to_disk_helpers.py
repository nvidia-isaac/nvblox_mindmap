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
import pickle

from nvblox_torch.mapper import Mapper
import torch
import zstandard

from mindmap.mapping.helpers.nvblox_output_helpers import get_vertices_and_features
from mindmap.mapping.nvblox_mapper_constants import MAPPER_TO_ID, NvbloxMappingCfg


def save_feature_mesh_to_disk(
    mapper: Mapper,
    mapping_config: NvbloxMappingCfg,
    num_excess_features: int,
    frame_index: int,
    save_directory: str,
    include_dynamic: bool,
):
    """
    Save the feature vertices.

    Args:
        vertices (torch.Tensor): The vertex coordinates of shape (N, 3).
        features (torch.Tensor): The feature vectors of shape (N, F) where F is the feature channel length.
        save_directory (str): Directory where the output file will be saved.
        index (int): Index of the current frame being processed.
        output_file_base_name (str): Base name of the output file to be created.
    """
    vertex_features_output_file_name = "nvblox_vertex_features.zst"
    assert not include_dynamic, "Dynamics are not supported for mesh encoding yet."
    vertices, features, _ = get_vertices_and_features(
        mapper,
        MAPPER_TO_ID.STATIC,
        mapping_config,
        remove_zero_features=True,
        num_excess_features=num_excess_features,
        sample_vertices=False,
    )

    assert vertices.shape[0] == features.shape[0]
    assert vertices.shape[1] == 3

    pc_ob = {
        "vertices": vertices.to(torch.float16).cpu(),
        "features": features.to(torch.float16).cpu(),
        "channel_length": features.shape[1],
    }

    # Pickle and compress with zst (a format that is fast to decompress).
    compressor = zstandard.ZstdCompressor(level=1)
    output_file_path = os.path.join(
        save_directory, f"{frame_index:04d}.{vertex_features_output_file_name}"
    )
    with open(output_file_path, "wb") as outfile:
        outfile.write(compressor.compress(pickle.dumps(pc_ob, protocol=pickle.HIGHEST_PROTOCOL)))

    return vertices, features


def save_serialized_nvblox_map_to_disk(
    mapper: Mapper,
    save_directory: str,
    index: int,
    include_dynamic: bool,
) -> None:
    """
    Save the serialized nvblox map to disk.

    Args:
        mapper: The mapper to save the map for.
        save_directory: The directory to save the map to.
        index: The index of the current frame being processed.
        include_dynamic: Whether to include the dynamic map in the output.
    """
    STATIC_NVBLOX_MAP_OUTPUT_FILE_NAME = "nvblox_map_static.nvblx"
    DYNAMIC_NVBLOX_MAP_OUTPUT_FILE_NAME = "nvblox_map_dynamic.nvblx"
    output_path = os.path.join(save_directory, f"{index:04d}.{STATIC_NVBLOX_MAP_OUTPUT_FILE_NAME}")
    mapper.save_map(output_path, MAPPER_TO_ID.STATIC)
    if include_dynamic:
        output_path = os.path.join(
            save_directory, f"{index:04d}.{DYNAMIC_NVBLOX_MAP_OUTPUT_FILE_NAME}"
        )
        mapper.save_map(output_path, MAPPER_TO_ID.DYNAMIC)

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import Dict, Tuple

from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import (
    BlockMemoryPoolParams,
    MapperParams,
    ProjectiveIntegratorParams,
    TsdfDecayIntegratorParams,
    ViewCalculatorParams,
)
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.timer import Timer
import torch
import torch.nn.functional as F

from mindmap.image_processing.feature_extraction import FeatureExtractor
from mindmap.image_processing.image_mask_operations import erode_mask, get_border_mask
from mindmap.mapping.nvblox_mapper_constants import MAPPER_TO_ID, NvbloxMappingCfg


def get_nvblox_mapper(mapper_config: NvbloxMappingCfg) -> Mapper:
    """
    Create and return an NVblox Mapper instance based on the provided configuration.

    Args:
        mapper_config (NvbloxMappingCfg): Configuration containing mapping parameters,

    Returns:
        Mapper: An instance of the NVblox Mapper class.
    """
    projective_integrator_params = ProjectiveIntegratorParams()
    projective_integrator_params.projective_integrator_max_integration_distance_m = (
        mapper_config.projective_integrator_max_integration_distance_m
    )
    projective_integrator_params.projective_appearance_integrator_measurement_weight = (
        mapper_config.projective_appearance_integrator_measurement_weight
    )

    tsdf_decay_integrator_params = TsdfDecayIntegratorParams()
    tsdf_decay_integrator_params.tsdf_decay_factor = mapper_config.tsdf_decay_factor

    view_calculator_params = ViewCalculatorParams()
    view_calculator_params.raycast_subsampling_factor = 1
    view_calculator_params.workspace_bounds_type = "kBoundingBox"
    view_calculator_params.workspace_bounds_min_corner_x_m = mapper_config.aabb_min_m[0]
    view_calculator_params.workspace_bounds_min_corner_y_m = mapper_config.aabb_min_m[1]
    view_calculator_params.workspace_bounds_min_height_m = mapper_config.aabb_min_m[2]
    view_calculator_params.workspace_bounds_max_corner_x_m = mapper_config.aabb_max_m[0]
    view_calculator_params.workspace_bounds_max_corner_y_m = mapper_config.aabb_max_m[1]
    view_calculator_params.workspace_bounds_max_height_m = mapper_config.aabb_max_m[2]

    # Reduce memory consumption
    block_memory_pool_params = BlockMemoryPoolParams()
    block_memory_pool_params.expansion_factor = 1.0
    block_memory_pool_params.num_preallocated_blocks = 0

    mapper_params = MapperParams()
    mapper_params.set_projective_integrator_params(projective_integrator_params)
    mapper_params.set_tsdf_decay_integrator_params(tsdf_decay_integrator_params)
    mapper_params.set_view_calculator_params(view_calculator_params)
    mapper_params.set_block_memory_pool_params(block_memory_pool_params)

    return Mapper(
        voxel_sizes_m=[mapper_config.voxel_size_m, mapper_config.voxel_size_m],
        integrator_types=[ProjectiveIntegratorType.TSDF, ProjectiveIntegratorType.TSDF],
        mapper_parameters=mapper_params,
    )


def nvblox_integrate(
    mapper: Mapper,
    nvblox_mapping_config: NvbloxMappingCfg,
    feature_extractor: FeatureExtractor,
    depth_frame: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pose: torch.Tensor,
    rgb: torch.Tensor,
    dynamic_mask: torch.Tensor,
    include_dynamic: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Integrate RGB, depth, and feature data from a camera frame into the nvblox mapper.

    Args:
        mapper (Mapper): The nvblox mapper instance for 3D reconstruction
        nvblox_mapping_config (NvbloxMappingCfg): Configuration containing mapping parameters,
        feature_extractor (FeatureExtractor): Extractor for computing image features
        depth_frame (torch.Tensor): Depth image of shape [256, 256] and dtype float32
        intrinsics (torch.Tensor): Camera intrinsics matrix of shape [3, 3] on CPU
        camera_pose (torch.Tensor): Camera pose matrix of shape [4, 4] on CPU
        rgb (torch.Tensor): RGB image of shape [256, 256, 3] and dtype uint8
        dynamic_mask (torch.Tensor): Mask of shape [256, 256] for dynamic objects valued as boolean
        include_dynamic (bool): Whether to include dynamic objects in the integration
    Returns:
        Dict[str, Dict[str, torch.Tensor]]: Nested dictionary containing integration images:
            - First level key: Mapper name ('STATIC' or 'DYNAMIC')
            - Second level contains the integration images:
                - depth_frame: Depth image used for integration
                - depth_mask: Binary mask for valid depth values
                - rgb_frame: RGB image used for integration
                - rgb_mask: Binary mask for RGB integration
                - feature_frame: Extracted features
                - feature_mask: Binary mask for feature integration
    """
    assert dynamic_mask.dtype == torch.bool

    # Create the static mask.
    if nvblox_mapping_config.use_dynamic_mask:
        static_mask = ~dynamic_mask
    else:
        static_mask = torch.ones_like(dynamic_mask).to(torch.bool)
    assert static_mask.dtype == torch.bool

    # Extract features
    with Timer("nvblox_mapper/compute_features"):
        feature_frame = feature_extractor.compute(rgb=rgb.unsqueeze(0)).squeeze()
    assert feature_frame.shape[:2] == nvblox_mapping_config.upscaled_feature_image_size

    nvblox_integration_images = {}
    static_nvblox_integration_images = integrate_frame(
        mapper=mapper,
        nvblox_mapping_config=nvblox_mapping_config,
        depth_frame=depth_frame,
        feature_frame=feature_frame,
        intrinsics=intrinsics,
        camera_pose=camera_pose,
        rgb=rgb,
        input_mask=static_mask,
        input_mask_erosion_iterations=nvblox_mapping_config.static_mask_erosion_iterations,
        valid_depth_mask_erosion_iterations=nvblox_mapping_config.valid_depth_mask_erosion_iterations,
        mapper_id=MAPPER_TO_ID.STATIC,
    )
    nvblox_integration_images[MAPPER_TO_ID.STATIC.name] = static_nvblox_integration_images

    if include_dynamic:
        dynamic_nvblox_integration_images = integrate_frame(
            mapper=mapper,
            nvblox_mapping_config=nvblox_mapping_config,
            depth_frame=depth_frame,
            feature_frame=feature_frame,
            intrinsics=intrinsics,
            camera_pose=camera_pose,
            rgb=rgb,
            input_mask=dynamic_mask,
            input_mask_erosion_iterations=nvblox_mapping_config.dynamic_mask_erosion_iterations,
            valid_depth_mask_erosion_iterations=nvblox_mapping_config.valid_depth_mask_erosion_iterations,
            mapper_id=MAPPER_TO_ID.DYNAMIC,
        )
        nvblox_integration_images[MAPPER_TO_ID.DYNAMIC.name] = dynamic_nvblox_integration_images

    return nvblox_integration_images


def integrate_frame(
    mapper: Mapper,
    nvblox_mapping_config: NvbloxMappingCfg,
    depth_frame: torch.Tensor,
    feature_frame: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pose: torch.Tensor,
    rgb: torch.Tensor,
    input_mask: torch.Tensor,
    input_mask_erosion_iterations: int,
    valid_depth_mask_erosion_iterations: int,
    mapper_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate a single frame of depth, color and feature data into the nvblox mapper.

    Args:
        mapper (Mapper): The nvblox mapper instance
        nvblox_mapping_config (NvbloxMappingCfg): Configuration for mapping parameters
        depth_frame (torch.Tensor): Depth image of shape [H, W]
        feature_frame (torch.Tensor): Feature image of shape [H, W, C]
        intrinsics (torch.Tensor): Camera intrinsics matrix of shape [3, 3]
        camera_pose (torch.Tensor): Camera pose matrix of shape [4, 4]
        rgb (torch.Tensor): RGB image of shape [H, W, 3] and dtype uint8
        input_mask (torch.Tensor): Binary mask of shape [H, W] indicating valid regions
        input_mask_erosion_iterations (int): Number of iterations to erode the inputmask
        valid_depth_mask_erosion_iterations (int): Number of iterations to erode the close object mask
        mapper_id (int): ID of the mapper to integrate into

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing the integrated images and masks:
            - depth_frame: Depth frame integrated into the mapper
            - depth_mask: Mask used to integrate the depth frame
            - rgb_frame: RGB frame integrated into the mapper
            - rgb_mask: Mask used to integrate the RGB frame
            - feature_frame: Feature frame integrated into the mapper
            - feature_mask: Mask used to integrate the feature frame
    """
    assert input_mask.dtype == torch.bool

    valid_depth_mask = depth_frame > nvblox_mapping_config.min_integration_distance_m
    assert valid_depth_mask.dtype == torch.bool

    depth_mask = torch.logical_and(input_mask, valid_depth_mask)
    assert depth_mask.dtype == torch.bool

    mapper.add_depth_frame(
        depth_frame, camera_pose.cpu(), intrinsics.cpu(), depth_mask.to(torch.uint8), mapper_id
    )

    # Color Nvblox integration
    mapper.add_color_frame(
        rgb.contiguous(),
        camera_pose.cpu(),
        intrinsics.cpu(),
        mask_frame=depth_mask.to(torch.uint8),
        mapper_id=mapper_id,
    )

    # Expand the masks to account for bleeding features due to convolution.
    # Doing this separately for the input mask and the valid depth mask (different number of iterations).
    input_mask_eroded = erode_mask(input_mask, iterations=input_mask_erosion_iterations)
    valid_depth_mask_eroded = erode_mask(
        valid_depth_mask, iterations=valid_depth_mask_erosion_iterations
    )
    depth_mask_eroded = torch.logical_and(input_mask_eroded, valid_depth_mask_eroded)

    # Scale the intrinsics.
    # Note: we're only scaling the first two rows since a calibration matrix should always have last row [0,0,1]
    assert feature_frame.shape[0] == feature_frame.shape[1], "We're only supporting square images"
    assert rgb.shape[0] == rgb.shape[1], "We're only supporting square images"
    upscale_factor = feature_frame.shape[0] / rgb.shape[0]
    feature_intrinsics = intrinsics
    feature_intrinsics[:2, :] *= upscale_factor

    # Upscale the masks to the feature frame size.
    depth_mask_eroded_upscaled = (
        F.interpolate(
            depth_mask_eroded.unsqueeze(0).unsqueeze(0).to(torch.uint8),
            size=(feature_frame.shape[0], feature_frame.shape[1]),
            mode="nearest",  # Use nearest neighbor to preserve binary value
        )
        .squeeze(0)
        .squeeze(0)
        .to(torch.bool)
    )

    # The clip feature image contains artifacts on the border which we exclude.
    border_mask = get_border_mask(
        feature_frame.shape, nvblox_mapping_config.feature_mask_border_percent, feature_frame.device
    )[0]

    feature_mask = torch.logical_and(border_mask, depth_mask_eroded_upscaled).to(torch.uint8)

    mapper.add_feature_frame(
        feature_frame.contiguous().to(dtype=torch.float16),
        camera_pose.cpu(),
        feature_intrinsics.cpu(),
        feature_mask,
        mapper_id,
    )

    nvblox_integration_images = {
        "depth_frame": depth_frame,
        "depth_mask": depth_mask,
        "rgb_frame": rgb.permute(2, 0, 1) / 255.0,
        "rgb_mask": depth_mask,
        "feature_frame": feature_frame,
        "feature_mask": feature_mask,
        "input_mask": input_mask,
    }

    return nvblox_integration_images

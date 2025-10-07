# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from copy import deepcopy
from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class DepthNoiseCfg:
    noise_scale_m: float = 0.02
    """This is the standard deviation noise on the depth camera values at 1m."""

    maximum_depth_m: float = 1.5
    """The maximum depth we consider to be valid."""

    p_dropout: float = 0.003
    """Probability that each depth pixel will be replaced with a zero."""

    p_random_depth: float = 0.003
    """Probability that each depth pixel will be replaced with a random value between 0.0 and maximum_depth."""

    max_num_bars: int = 5
    """The maximum number of bars that max randomly appear."""

    bar_thickness_px: int = 4
    """The thickness of the random bars in pixels."""

    bar_length_px: int = 30
    """The length of the random bars in pixels."""

    baseline_px: int = 100
    """This value if related to the baseline and determines quantization.

    Lower means more quantization.
    See discussion of the baseline value here:
    https://github.com/ankurhanda/simkinect/blob/master/camera_utils.py#L231
    """


def get_noised_depth_image(
    depth_image: torch.Tensor, depth_noise_config: DepthNoiseCfg
) -> torch.Tensor:
    noise = get_depth_noise(depth_image, depth_noise_config)
    depth = do_depth_quantization(depth_image, depth_noise_config)
    depth_noised = depth + noise
    depth_noised = do_dropout_and_replacement(depth_noised, depth_noise_config)
    depth_noised = add_random_bars_to_image(depth_noised, depth_noise_config)
    return depth_noised


def get_depth_not_valid_mask(
    depth_image: torch.Tensor, depth_noise_config: DepthNoiseCfg
) -> torch.Tensor:
    # Depth is not valid if:
    # - == inf
    # - >depth_noise_config.maximum_depth_m
    inf_flags = torch.isinf(depth_image)
    beyond_max_depth_flags = depth_image > depth_noise_config.maximum_depth_m
    not_valid_flags = torch.logical_or(inf_flags, beyond_max_depth_flags)
    return not_valid_flags


def get_depth_noise(depth_image: torch.Tensor, depth_noise_config: DepthNoiseCfg) -> torch.Tensor:
    # Generate noise proportional to square of the depth
    noise = torch.normal(
        mean=torch.zeros_like(depth_image),
        std=depth_noise_config.noise_scale_m * torch.square(depth_image),
    )
    # Zero the noise where the depth isn't valid
    not_valid_flags = get_depth_not_valid_mask(depth_image, depth_noise_config)
    noise[not_valid_flags] = 0.0
    return noise


def do_dropout_and_replacement(
    depth_image: torch.Tensor, depth_noise_config: DepthNoiseCfg
) -> torch.Tensor:
    # Copy
    depth_noised = deepcopy(depth_image)
    # Dropout
    dropout_probabilities = depth_noise_config.p_dropout * torch.ones_like(depth_image)
    dropout_flags = torch.bernoulli(dropout_probabilities) == 1.0
    depth_noised[dropout_flags] = 0.0
    # Replacement
    random_probabilities = depth_noise_config.p_random_depth * torch.ones_like(depth_image)
    random_flags = torch.bernoulli(random_probabilities) == 1.0
    random_depths = torch.rand_like(depth_image) * depth_noise_config.maximum_depth_m
    depth_noised[random_flags] = random_depths[random_flags]
    return depth_noised


def add_random_bars_to_image(
    depth_image: torch.Tensor, depth_noise_config: DepthNoiseCfg
) -> torch.Tensor:
    # NOTE(alexmillane): At the moment the bars added can be beyond the depth in the depth map.
    # This generates a hole in the depth map. Probably in the future we should take the minimum
    # between the depth map and the bar image.
    assert depth_image.dim() == 3
    assert depth_image.shape[2] == 1
    # Convert to numpy and 3D
    depth_image_np = torch.squeeze(deepcopy(depth_image)).cpu().numpy()

    # Image size
    width = depth_image_np.shape[1]
    height = depth_image_np.shape[0]

    # Bar prior to random transform
    num_bars = np.round(np.random.uniform(low=0, high=depth_noise_config.max_num_bars + 1)).astype(
        int
    )
    half_length = depth_noise_config.bar_length_px // 2
    half_thickness = depth_noise_config.bar_thickness_px // 2
    vertices_B = np.array(
        [
            [-half_length, -half_thickness],
            [+half_length, -half_thickness],
            [+half_length, +half_thickness],
            [-half_length, +half_thickness],
        ]
    )

    for _ in range(num_bars):
        # Random depth, angle, and position
        depth = np.random.uniform(low=0.0, high=depth_noise_config.maximum_depth_m)
        angle = np.random.uniform(low=0.0, high=np.pi)
        t = (
            np.random.uniform(low=np.array([0, 0]), high=np.array([width, height]))
            .astype(int)
            .reshape((2, 1))
        )
        # Transform
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        # Vertices in the image frame
        vertices_I = np.round((R @ vertices_B.T + t).T).astype(int)
        # Paint
        cv2.fillPoly(depth_image_np, [vertices_I], depth)
    return torch.tensor(depth_image_np, device=depth_image.device).reshape(depth_image.shape)


def do_depth_quantization(
    depth_image: torch.Tensor, depth_noise_config: DepthNoiseCfg
) -> torch.Tensor:
    # See discussion of the baseline value here:
    # https://github.com/ankurhanda/simkinect/blob/master/camera_utils.py#L231
    depth_quantized = depth_noise_config.baseline_px / torch.round(
        depth_noise_config.baseline_px / depth_image + 0.5
    )
    return depth_quantized

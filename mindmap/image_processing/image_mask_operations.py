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
import torch.nn.functional as F


def erode_mask(mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """Erodes a mask by max pooling the inverted mask.

    Args:
        mask (torch.Tensor): Binary mask of shape (H, W) on CUDA.
        kernel_size (int): Size of the dilation kernel (should be odd).
        iterations (int): Number of expansion iterations.

    Returns:
        torch.Tensor: Expanded binary mask where zeros have been expanded.
    """
    assert mask.dim() == 2, "Mask must be 2D"
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    assert mask.dtype == torch.bool, "Mask must be of type bool"
    inverted_mask = ~mask
    pad = (kernel_size - 1) // 2
    for _ in range(iterations):
        # Expand the zero regions by performing max pooling
        expanded = F.max_pool2d(
            inverted_mask.to(torch.float).unsqueeze(0).unsqueeze(0),
            kernel_size,
            stride=1,
            padding=pad,
        )
        inverted_mask = expanded.to(torch.bool).squeeze(0).squeeze(0)
    return ~inverted_mask


def get_border_mask(
    mask_shape: torch.Size, mask_border_percent: float, device: torch.device
) -> Tuple[torch.Tensor, bool, bool]:
    """
    Create a mask for the given tensor with a specified border percentage set to zero.

    Args:
        mask_shape (torch.Size): The shape of the mask tensor.
        mask_border_percent (float): The percentage of the border to be masked (set to zero).
        device (torch.device): The device to create the mask on.
    Returns:
        torch.Tensor: The mask tensor with borders set to False.
        int: The height of the border in pixels.
        int: The width of the border in pixels.
    """
    height, width = mask_shape[:2]
    mask = torch.full((height, width), True, dtype=torch.bool, device=device)
    border_h = int(mask_border_percent * 0.01 * height)
    border_w = int(mask_border_percent * 0.01 * width)
    if border_h > 0 and border_w > 0:
        mask[:border_h, :] = False
        mask[-border_h:, :] = False
        mask[:, :border_w] = False
        mask[:, -border_w:] = False
    return mask, border_h, border_w


def downscale_mask(mask: torch.Tensor, downscale_factor: int) -> torch.Tensor:
    """Downscales a mask by the given factor.

    The downscaling is perfomed by pooling under logical AND. Hence, a downscaled pixel will be
    active only if all pixels contributing to its value are active.

    Image width and height must be divisible with downscale_factor.

    Args:
        mask (torch.Tensor): Binary mask of shape (B, 1, H, W) on CUDA.
        downscale_factor (int): The factor to downscale the mask by.
    """
    assert downscale_factor > 0, "Downscale factor must be positive"
    assert mask.dim() == 4, "Mask must be 4D"
    assert mask.dtype == torch.bool, "Mask must be of type bool"
    assert mask.shape[2] % downscale_factor == 0, "Mask width must be divisible by downscale factor"
    assert (
        mask.shape[3] % downscale_factor == 0
    ), "Mask height must be divisible by downscale factor"
    # We create a view of the mask with two extra dimensions that are reduced with "all" in order to achive &-pooling
    view = mask.view(
        (
            mask.shape[0],
            mask.shape[1],
            int(mask.shape[2] / downscale_factor),
            downscale_factor,
            int(mask.shape[3] / downscale_factor),
            downscale_factor,
        )
    )
    return torch.all(torch.all(view, dim=-1), dim=-2)

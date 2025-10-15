# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import torch


def convert_rgb_to_model_input(image: torch.Tensor) -> torch.Tensor:
    """Converts an RGB image tensor to the format expected by the model.

    Takes an RGB image tensor with channels last and converts it to channels first format.
    Normalizes pixel values from [0,255] to [0,1] range.

    Args:
        image (torch.Tensor): Input RGB image tensor of shape (H,W,3) in range [0,255]

    Returns:
        torch.Tensor: Converted image tensor of shape (3,H,W) in float32 format in range [0,1]
    """
    assert image.dim() == 3
    assert image.shape[-1] == 3  # channel dimension

    # Convert range to [0, 1]
    assert image.min() >= 0
    assert image.max() >= 0
    image = image / 255.0
    assert image.min() >= 0
    assert image.max() <= 1

    # Move channel dimension to front
    image = image.permute(2, 0, 1).type(torch.float32)

    return image

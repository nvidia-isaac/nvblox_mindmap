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


def assert_in_bounds_of_type(tensor: torch.Tensor, type: torch.dtype) -> None:
    """Asserts that all elements of the tensor are within the bounds of the range
       representable by the type.

    Args:
        tensor (torch.Tensor): The tensor containing the values to be checked.
        type (torch.dtype): The type whose bounds we want to respect.
    """
    assert torch.min(tensor.flatten()) >= torch.iinfo(type).min, "Tried to convert out of bounds."
    assert torch.max(tensor.flatten()) <= torch.iinfo(type).max, "Tried to convert out of bounds."

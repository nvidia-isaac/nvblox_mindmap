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

from mindmap.image_processing.image_mask_operations import downscale_mask


def test_downscale_mask_random():
    mask = torch.randint(0, 2, (2, 1, 64, 64), dtype=torch.bool, device="cuda")
    downscale_factor = 4
    downscaled_mask = downscale_mask(mask, downscale_factor)
    assert downscaled_mask.shape == (2, 1, 16, 16)


def test_downscale_mask_known_pattern():
    mask = torch.zeros((1, 1, 4, 4), dtype=torch.bool, device="cuda")

    # All pixels in each 2x2 blocks needs to be true for the downscaled mask to be true
    mask[0, 0, :, :] = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 2], [1, 0, 1, 1]], dtype=torch.bool, device="cuda"  #
    )
    downscale_factor = 2
    downscaled_mask = downscale_mask(mask, downscale_factor)

    assert downscaled_mask.shape == (1, 1, 2, 2)

    # Expected True if any of the elements a in a 2x2 block is true, otherwise False
    assert downscaled_mask[:, :, 0, 0] == True
    assert downscaled_mask[:, :, 0, 1] == False
    assert downscaled_mask[:, :, 1, 0] == False
    assert downscaled_mask[:, :, 1, 1] == False

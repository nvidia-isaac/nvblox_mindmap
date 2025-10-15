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

import einops
import torch
from torch.nn import functional as F


def get_pcd_for_visualization(pointcloud: torch.Tensor, image_dim: Tuple[int, int]):
    """Upscale a pointcloud to the desired image dimension."""
    pcd = F.interpolate(
        einops.rearrange(pointcloud.unsqueeze(0), "b h w c -> b c h w"), image_dim, mode="bilinear"
    )
    pcd = einops.rearrange(pcd, "b c h w -> b h w c")
    return pcd

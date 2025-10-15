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

from mindmap.visualization.color_maps.color_map_green_pink_tones import get_pink_green_color_map


class ColorMap(object):
    """
    Utility class to load a color map and colorize a tensor
    """

    def __init__(self):
        # Get the pink/green color map
        self.color_map = get_pink_green_color_map()

    def colorize(self, x: torch.Tensor, value_range: Tuple[float, float] = None):
        """
        Colorizes a rank 2 tensor based on the color map. If 'value_range' is given values in the
        tensor are first mapped from value_range to (0, 1). Out of bounds values are clamped.
        """
        if value_range is not None:
            x = (x - value_range[0]) / (value_range[1] - value_range[0])
        x = torch.clamp(x, min=0.0, max=1.0)
        if self.color_map is None:
            return x
        x = self.color_map.to(x.device)[(x * 255).long()]
        return x.permute(2, 0, 1)

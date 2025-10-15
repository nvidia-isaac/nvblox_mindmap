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
import torchvision
import wandb

from mindmap.visualization.color_maps.color_map import ColorMap


class TensorVisualizer(object):
    """
    A helper class for visualizing tensors with wandb
    """

    def __init__(self):
        self._vis = {}
        self.enabled = False
        self.color_map = ColorMap()

    def register_tensor(
        self, tag: str, shape: torch.Size, nrow: int = 1, scale_factor: int = 1, value_range=None
    ):
        """
        Call for each channel which you want to visualize.
        """
        self._vis[tag] = {
            "tensor": torch.zeros(shape).cuda(),
            "nrow": nrow,
            "scale_factor": scale_factor,
            "value_range": value_range,
        }

    def enable(self):
        """
        Sets the enabled flat to true and also zeros all tensors
        """
        self.enabled = True
        for tag, item in self._vis.items():
            self._vis[tag]["tensor"] = torch.zeros(self._vis[tag]["tensor"].shape).cuda()

    def disable(self):
        """
        Sets the enabled flag to false
        """
        self.enabled = False

    def set(
        self,
        tag: str,
        tensor: torch.tensor,
        value_range: Tuple[float, float] = None,
        scale_factor: int = 1,
    ):
        """
        Sets a tensor which was previously initialized with init
        """
        self._vis[tag]["tensor"] = tensor.detach()
        # Update data normalization / scaling if needed
        if value_range is not None:
            self._vis[tag]["value_range"] = value_range
        self._vis[tag]["scale_factor"] = scale_factor

    def log_tensor_to_wandb(self, iter: int, prefix: str = ""):
        """
        Shows all tensors which have been set to visualize in wandb
        """
        for tag, item in self._vis.items():
            self._show_channel(iter, prefix + tag, item)

    def _show_channel(self, iter: int, tag: str, item):
        """
        Internal helper function to show one channel
        """
        x = item["tensor"]
        self._show_tensor(
            iter,
            tag,
            x,
            value_range=item["value_range"],
            nrow=item["nrow"],
            scale_factor=item["scale_factor"],
        )

    def _show_tensor(
        self,
        step_id: int,
        tag: str,
        x: torch.tensor,
        value_range: Tuple[float, float],
        nrow: int,
        scale_factor: int = None,
    ):
        """
        Internal helper function to show one tensor
        """
        x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        if x.shape[-3] == 1:
            # Use color map for single channel images
            grid = torchvision.utils.make_grid(x, nrow=nrow, normalize=True, scale_each=True)
            grid = self.color_map.colorize(grid[0], value_range=value_range)
        else:
            # Show multi channel images directly
            grid = torchvision.utils.make_grid(x)
        if scale_factor is not None:
            grid = F.interpolate(grid, scale_factor=scale_factor)
        # accumulate until logging loss
        # TODO: replace step with step_id instead of wandb native step
        wandb.log({tag: wandb.Image(grid)}, commit=False)

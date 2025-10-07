# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import Any, List

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def get_axis_markers(marker_names: List[str], prim_path: str) -> VisualizationMarkers:
    """Gets a set of axis markers for the given marker names and prim path."""
    scale = (0.1, 0.1, 0.1)
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    markers = {}
    for marker_name in marker_names:
        markers[marker_name] = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=scale,
        )
    frame_marker_cfg = VisualizationMarkersCfg(markers=markers, prim_path=prim_path)
    return VisualizationMarkers(frame_marker_cfg)

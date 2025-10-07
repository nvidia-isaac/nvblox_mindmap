# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, fields
from typing import Dict

from mindmap.isaaclab_utils.isaaclab_camera_handler import IsaacLabCameraHandler


@dataclass
class ObservationBase(ABC):
    """Observation base class."""

    """Common helper: return every attribute that is a camera handler."""

    def get_cameras(self) -> Dict[str, "IsaacLabCameraHandler"]:
        """
        Collect every attribute that currently holds an `IsaacLabCameraHandler`
        (e.g. `table_camera`, `wrist_camera`, …) and return them in a dict.

        Example
        -------
        >>> obs = ArmEmbodimentObservation(...)
        >>> obs.cameras()
        {'table_camera': <IsaacLabCameraHandler …>,
         'wrist_camera': <IsaacLabCameraHandler …>}
        """
        result: Dict[str, IsaacLabCameraHandler] = {}
        for f in fields(self):  # iterate over *all* dataclass fields
            val = getattr(self, f.name)
            if isinstance(val, IsaacLabCameraHandler):
                result[f.name] = val
        return result

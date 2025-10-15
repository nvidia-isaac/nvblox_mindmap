# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from enum import Enum


class RenderSettings(Enum):
    # Use default settings from Isaac-Lab
    DEFAULT = "default"
    # Aims to create reproducible results accross platforms, but might not be entirely deterministic. Reduces image quality.
    DETERMINISTIC = "deterministic"
    # Highest quality. Uses path tracing for rendering.
    HIGH_QUALITY = "high_quality"

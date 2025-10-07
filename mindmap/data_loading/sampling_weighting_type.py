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


class SamplingWeightingType(Enum):
    """Defines the types of sampling weighting."""

    UNIFORM = 0
    GRIPPER_STATE_CHANGE = 1
    NONE = 3  # Use for pure sequential processing


def get_sampling_weighting_type(weighting_type: str) -> SamplingWeightingType:
    try:
        return SamplingWeightingType[weighting_type.upper()]
    except KeyError:
        raise ValueError(f"'{weighting_type}' is not a valid member of SamplingWeightingType")

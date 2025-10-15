# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

from tap import Tap
import torch

from mindmap.tasks.tasks import Tasks

# Scale factor to save float32 depth as uint16 in Mimicgen.
DEPTH_SCALE_FACTOR = 1000.0

# Named mapping to access data in the Mimicgen data tensors.
CAMERA_NAME_TO_ID = {"table_rgb": 0, "wrist_rgb": 1}


# Named mapping to access the mapper id.
class MAPPER_TO_ID(int, Enum):
    STATIC = 0
    DYNAMIC = 1


COMMON_NVBLOX_MAPPER_CFG = {
    "projective_integrator_max_integration_distance_m": 5.0,
    "voxel_size_m": 0.01,
    "unobserved_value": 0.0,
    "required_tensor_shape_dict": {"x": 128, "y": 128, "z": 64},
    "upscaled_feature_image_size": (512, 512),
    "feature_mask_border_percent": 5,
    "static_mask_erosion_iterations": 17,
    "dynamic_mask_erosion_iterations": 3,
    "projective_appearance_integrator_measurement_weight": 1.0,
}

TASK_TO_NVBLOX_MAPPER_CFG = {
    Tasks.MUG_IN_DRAWER.name: {
        "tsdf_decay_factor": 0.999,
        "aabb_min_m": torch.tensor([-0.2, -0.8, -0.2]),
        "aabb_max_m": torch.tensor([0.9, 0.8, 1.0]),
        "min_integration_distance_m": 0.37,
        "use_dynamic_mask": True,
        "dynamic_class_labels": ["robot_arm"],
        "valid_depth_mask_erosion_iterations": 10,
    },
    Tasks.CUBE_STACKING.name: {
        "tsdf_decay_factor": 0.98,
        "aabb_min_m": torch.tensor([-0.25, -0.65, -0.07]),
        "aabb_max_m": torch.tensor([1.0, 0.62, 0.56]),
        "min_integration_distance_m": 0.10,
        "use_dynamic_mask": True,
        "dynamic_class_labels": ["robot_arm"],
        "valid_depth_mask_erosion_iterations": 20,
    },
    Tasks.DRILL_IN_BOX.name: {
        "tsdf_decay_factor": 0.98,
        "aabb_min_m": torch.tensor([-0.37, -0.75, -0.13]),
        "aabb_max_m": torch.tensor([0.95, 0.75, 0.65]),
        "min_integration_distance_m": 0.30,
        "use_dynamic_mask": True,
        "dynamic_class_labels": ["robot"],
        "valid_depth_mask_erosion_iterations": 20,
    },
    Tasks.STICK_IN_BIN.name: {
        "tsdf_decay_factor": 0.98,
        "aabb_min_m": torch.tensor([3.7, 1.5, 0.44]),
        "aabb_max_m": torch.tensor([5.5, 3.2, 1.25]),
        "min_integration_distance_m": 0.30,
        "use_dynamic_mask": True,
        "dynamic_class_labels": ["robot"],
        "valid_depth_mask_erosion_iterations": 20,
    },
}


def get_workspace_bounds(task: Tasks) -> torch.Tensor:
    """Return the workspace bounds as a 2x3 tensor given a task name"""
    task_cfg = TASK_TO_NVBLOX_MAPPER_CFG[task.name]
    return torch.stack([task_cfg["aabb_min_m"], task_cfg["aabb_max_m"]])


@dataclass
class NvbloxMappingCfg:
    """
    Configuration with parameters for the nvblox mapping.

    Args:
        task (str): The name of the task to be mapped. Based on this task name the
            parameters are set from the TASK_TO_NVBLOX_MAPPER_CFG dictionary.
    """

    args: Tap = None

    # Nvblox mapper parameters
    projective_integrator_max_integration_distance_m: float = None
    tsdf_decay_factor: float = None
    voxel_size_m: float = None
    # The minimum and maximum corners of the grid to be stored after mapping.
    # Note that his is not directly a mapper parameter, but rather applied to post processing.
    aabb_min_m: torch.Tensor = None
    aabb_max_m: torch.Tensor = None
    # Unobserved areas within the dense output tsdf and feature grid will be filled with that value.
    unobserved_value: float = None
    # Don't integrate depth closer than a threshold.
    min_integration_distance_m: float = None
    # Whether to use the static mask for integration or integrate the full frames.
    use_dynamic_mask: bool = None
    # List of class labels to consider as dynamic objects.
    dynamic_class_labels: List[str] = None

    # Output shapes of the tsdf and feature grid.
    required_tensor_shape_dict: Dict[str, int] = None

    # Desired size of feature image. These are upscaled from the output of the feature extractor.
    upscaled_feature_image_size: Tuple[int, int] = None

    # Mask the borders of the feature image by a percentage of the full image to get
    # rid of unwanted artifacts.
    feature_mask_border_percent: int = None

    # To account for bleeding features due to convolution we dilate the static and dynamic masks.
    # Each iteration is a dilation by 1 pixel in all directions.
    static_mask_erosion_iterations: int = None
    dynamic_mask_erosion_iterations: int = None
    valid_depth_mask_erosion_iterations: int = None

    # Measurement weight for the feature fusion.
    projective_appearance_integrator_measurement_weight: float = None

    def _maybe_override_arg(self, arg_name: str, arg_value: Any):
        """
        Override an argument if it is provided by the user.
        """
        if arg_value is not None:
            print(f"Overriding {arg_name} with user provided {arg_value}")
            setattr(self, arg_name, arg_value)

    def __post_init__(self):
        """
        Post-initialization method to set parameters based on the task name.
        """
        assert self.args is not None, "args must be provided."
        assert (
            self.args.task.name in TASK_TO_NVBLOX_MAPPER_CFG
        ), f"{self.args.task.name} is not a recognized task."
        print(f"Setting nvblox mapper parameters for task: {self.args.task.name}")
        for param_name, param_value in COMMON_NVBLOX_MAPPER_CFG.items():
            assert hasattr(self, param_name), f"{param_name} is not a recognized attribute."
            setattr(self, param_name, param_value)
        for param_name, param_value in TASK_TO_NVBLOX_MAPPER_CFG[self.args.task.name].items():
            assert hasattr(self, param_name), f"{param_name} is not a recognized attribute."
            setattr(self, param_name, param_value)
        for attr in self.__dict__:
            assert getattr(self, attr) is not None, f"{attr} must be provided."

        # Override args provided by the user
        self._maybe_override_arg("voxel_size_m", self.args.voxel_size_m)
        self._maybe_override_arg(
            "projective_appearance_integrator_measurement_weight",
            self.args.projective_appearance_integrator_measurement_weight,
        )

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import Dict, Tuple

import torch

from mindmap.data_loading.data_types import DataType, includes_mesh
from mindmap.image_processing.feature_extraction import FeatureExtractorType, get_feature_extractor
from mindmap.isaaclab_utils.isaaclab_camera_handler import IsaacLabCameraHandler
from mindmap.mapping.helpers.nvblox_input_helpers import (
    get_nvblox_inputs_from_camera_handler,
    get_nvblox_inputs_from_sample,
)
from mindmap.mapping.helpers.nvblox_mapping_helpers import (
    NvbloxMappingCfg,
    get_nvblox_mapper,
    nvblox_integrate,
)
from mindmap.mapping.helpers.nvblox_output_helpers import get_vertices_and_features
from mindmap.mapping.helpers.nvblox_to_disk_helpers import (
    save_feature_mesh_to_disk,
    save_serialized_nvblox_map_to_disk,
)
from mindmap.mapping.nvblox_mapper_constants import CAMERA_NAME_TO_ID, MAPPER_TO_ID
from mindmap.visualization.utils import get_pcd_for_visualization


class IsaacLabNvbloxMapper:
    """
    Integrates IsaacLab data into nvblox maps.
    """

    def __init__(self, mapping_data_type: DataType, args, device: str) -> None:
        """
        Initialize the IsaacLabNvbloxMapper.
        """
        self.mapping_data_type = mapping_data_type
        self.include_dynamic = args.include_dynamic
        self.num_vertices_to_sample = args.num_vertices_to_sample
        self.vertex_sampling_method = args.vertex_sampling_method
        self.save_serialized_nvblox_map_to_disk = args.save_serialized_nvblox_map_to_disk
        self.device = device
        self.mapping_config = NvbloxMappingCfg(args)
        self.mapper = get_nvblox_mapper(self.mapping_config)

        if self.mapping_data_type == DataType.MESH and self.include_dynamic:
            raise ValueError("Dynamics are not supported for mesh generation yet.")

        # Feature extractor
        if args.feature_type == FeatureExtractorType.CLIP_RESNET50_FPN:
            assert (
                args.fpn_checkpoint is not None
            ), "FPN checkpoint required for mapping with CLIP extractor"
        self.feature_extractor = get_feature_extractor(
            feature_extractor_type=args.feature_type,
            pad_to_nvblox_dim=True,
            desired_output_size=self.mapping_config.upscaled_feature_image_size,
            fpn_path=args.fpn_checkpoint,
        )

        # Save the last nvblox_integration_images for visualization.
        self.last_nvblox_integration_images = {}

    def update_reconstruction_from_camera(self, camera_handler: IsaacLabCameraHandler) -> None:
        """
        Update the nvblox reconstruction with the latest data from IsaacLab.
        """
        (
            depth_frame,
            intrinsics,
            camera_pose,
            rgb,
            dynamic_mask,
            pointcloud,
        ) = get_nvblox_inputs_from_camera_handler(
            camera_handler, self.mapping_config.dynamic_class_labels
        )

        self._update_reconstruction(
            depth_frame,
            intrinsics,
            camera_pose,
            rgb,
            dynamic_mask,
            pointcloud,
            camera_handler.camera_name,
        )

    def update_reconstruction_from_sample(
        self, sample: Dict[str, torch.Tensor], camera_name: str
    ) -> None:
        """
        Update the nvblox reconstruction with the latest data from a sample.
        """
        num_cams = sample["depths"].shape[1]
        if num_cams == 1:
            camera_index = 0
        else:
            camera_index = CAMERA_NAME_TO_ID[camera_name]

        (
            depth_frame,
            intrinsics,
            camera_pose,
            rgb,
            dynamic_mask,
            pointcloud,
        ) = get_nvblox_inputs_from_sample(sample, camera_index)

        self._update_reconstruction(
            depth_frame, intrinsics, camera_pose, rgb, dynamic_mask, pointcloud, camera_name
        )

    def _update_reconstruction(
        self,
        depth_frame: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_pose: torch.Tensor,
        rgb: torch.Tensor,
        dynamic_mask: torch.Tensor,
        pointcloud: torch.Tensor,
        camera_name: str,
    ) -> None:
        """
        Update the nvblox reconstruction with the latest camera data.

        This method integrates new camera data into the nvblox reconstruction
        by integrating depth, RGB, and feature data into the nvblox mapper

        Args:
            depth_frame (torch.Tensor): Depth image from camera, shape (H,W), dtype=float32
            intrinsics (torch.Tensor): Camera intrinsic matrix, shape (3,3), dtype=float32
            camera_pose (torch.Tensor): Camera pose as homogeneous transform, shape (4,4), dtype=float32
            rgb (torch.Tensor): RGB image, shape (H,W,3), dtype=uint8
            dynamic_mask (torch.Tensor): Boolean mask for dynamic objects, shape (H,W), dtype=bool
            pointcloud (torch.Tensor): World-frame pointcloud, shape (W,H,3), dtype=float32
            camera_name (str): Name of the camera providing the data
        """
        nvblox_integration_images = nvblox_integrate(
            mapper=self.mapper,
            nvblox_mapping_config=self.mapping_config,
            feature_extractor=self.feature_extractor,
            depth_frame=depth_frame,
            intrinsics=intrinsics,
            camera_pose=camera_pose,
            rgb=rgb,
            dynamic_mask=dynamic_mask,
            include_dynamic=self.include_dynamic,
        )

        # Add point cloud images used for visualization.
        for mapper_id in MAPPER_TO_ID:
            if mapper_id in nvblox_integration_images:
                nvblox_integration_images[mapper_id]["pcd"] = get_pcd_for_visualization(
                    pointcloud, self.mapping_config.upscaled_feature_image_size
                )

        self.last_nvblox_integration_images[camera_name] = nvblox_integration_images

    def save_nvblox_map_to_disk(
        self, frame_index: int, root_directory: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Save the current state of the nvblox map to disk and return visualization data.

        Saves the TSDF grid and feature grid tensors to compressed files. For mesh reconstructions,
        also extracts vertices and features. The saved files use the .zst compression format.

        Args:
            frame_index (int): Index of the current frame, used for naming the saved files
            root_directory (str): Directory where the map files will be saved

        Returns:
                - vertices: Vertex positions tensor if using mesh reconstruction, None otherwise
                - features: Vertex features tensor if using mesh reconstruction, None otherwise
        """
        features = None
        vertices = None
        num_excess_features = self.feature_extractor.num_excess_features()

        if includes_mesh(self.mapping_data_type):
            vertices, features = save_feature_mesh_to_disk(
                self.mapper,
                self.mapping_config,
                num_excess_features,
                frame_index,
                root_directory,
                self.include_dynamic,
            )

        if self.save_serialized_nvblox_map_to_disk:
            save_serialized_nvblox_map_to_disk(
                self.mapper,
                root_directory,
                frame_index,
                self.include_dynamic,
            )

        return vertices, features

    def get_nvblox_model_inputs(
        self, mapper_id: int, remove_zero_features: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get vertices and features from the nvblox mapper.

        Args:
            mapper_id (int): ID of the mapper to get vertices and features from.
            remove_zero_features (bool): Whether to remove vertices with zero features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - vertices: Tensor of vertex positions
                - features: Tensor of vertex features
        """
        samples = {
            "vertex_features": None,
            "vertices": None,
        }
        num_excess_features = self.feature_extractor.num_excess_features()

        if includes_mesh(self.mapping_data_type):
            (
                samples["vertices"],
                samples["vertex_features"],
                samples["vertices_valid_mask"],
            ) = get_vertices_and_features(
                self.mapper,
                mapper_id,
                self.mapping_config,
                remove_zero_features,
                num_excess_features,
                sample_vertices=True,
                number_of_vertices_to_sample=self.num_vertices_to_sample,
                vertex_sampling_method=self.vertex_sampling_method,
            )

            samples["vertex_features"] = (
                samples["vertex_features"].to(torch.float32).to(self.device)
            )
            samples["vertices"] = samples["vertices"].to(torch.float32).to(self.device)
        else:
            raise NotImplementedError(f"Invalid data type: {self.mapping_data_type}")

        return samples

    def clear(self):
        """Clears the nvblox mapper."""
        self.mapper.clear()

    def decay(self):
        """Decays the tsdf values in the mapper."""
        self.mapper.decay()

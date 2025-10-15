# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import Any, Dict, List, Tuple
import warnings

from isaaclab.utils import convert_dict_to_backend
import torch
from transforms3d.quaternions import quat2mat

from mindmap.image_processing.backprojection import get_camera_pointcloud


class IsaacLabCameraHandler:
    """
    Holds and processes RGB, depth, and semantic segmentation data from an IsaacLab camera object.

    Args:
        camera (Camera): The camera object containing sensor data
    """

    def __init__(self, camera: Any, camera_name: str):
        # NOTE(remos): The camera is of type isaaclab.sensors.Camera
        # but importing from isaaclab without running a simulation triggers an ImportError.
        # Therefore, type annotation is replaced with Any.
        self._camera = camera
        self._camera_name = camera_name

        # Assert required camera data keys exist
        required_keys = ["rgb", "distance_to_image_plane", "semantic_segmentation"]
        for key in required_keys:
            assert (
                key in self._get_camera_output().keys()
            ), f"Required camera data key '{key}' not found in camera output"

        self.DYNAMIC_SEGMENTATION_CLASS_ID = 0

    def _get_camera_output(self) -> Dict[str, torch.Tensor]:
        """
        Get the camera data dictionary.
        """
        return convert_dict_to_backend(self._camera.data.output, backend="torch")

    @property
    def camera(self) -> Any:
        """
        Get the camera object.
        """
        return self._camera

    @property
    def camera_name(self) -> str:
        return self._camera_name

    def get_intrinsics(self) -> torch.Tensor:
        """
        Extract camera intrinsic parameters from the camera object.

        Args:
            camera (Camera): The camera object

        Returns:
            torch.Tensor: Camera intrinsic matrix

        Raises:
            AssertionError: If more than one environment is detected
        """
        assert (
            self._camera.data.intrinsic_matrices.shape[0] == 1
        ), "We expect a single environment in Isaac Lab"
        return self._camera.data.intrinsic_matrices.data[0].clone()

    def get_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract camera pose information from the camera object.

        Args:
            camera (Camera): The camera object

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Camera translation and rotation quaternion

        Raises:
            AssertionError: If more than one environment is detected
        """
        assert self._camera.data.pos_w.shape[0] == 1, "We expect a single environment in Isaac Lab"
        translation = self._camera.data.pos_w.data[0].clone()
        rotation_quat = self._camera.data.quat_w_ros.data[0].clone()
        return translation, rotation_quat

    def get_pose_as_homo(self) -> torch.Tensor:
        """
        Convert the camera pose to a homogeneous transformation matrix.
        """
        translation, rotation_quat = self.get_pose()
        pose_transform = torch.eye(4, dtype=torch.float32)
        pose_transform[:3, :3] = torch.from_numpy(quat2mat(rotation_quat.cpu().numpy()))
        pose_transform[:3, 3] = translation
        return pose_transform

    def get_image_size(self) -> Tuple[int, int]:
        """
        Get the size of the image.
        """
        rgb_size = self.get_rgb().shape[:-1]
        depth_size = self.get_depth().shape
        semantic_size = self.get_semantic_segmentation()[0].shape[:-1]
        assert rgb_size == depth_size == semantic_size
        return rgb_size

    def get_rgb(self) -> torch.Tensor:
        """
        Extract RGB data from the camera data dictionary.

        Args:
            camera_data_dict (Dict): Dictionary containing camera data

        Returns:
            torch.Tensor: RGB image data in range [0, 255]

        Raises:
            AssertionError: If data format doesn't match expected shape or channels
        """
        color_data = self._get_camera_output()["rgb"].clone()
        assert color_data.shape[0] == 1, "We expect a single environment in Isaac Lab"
        assert color_data.shape[-1] == 3, "We expect a 3 channel color image"
        assert color_data.min() >= 0
        image_max = color_data.max()
        assert image_max <= 255
        if image_max == 0:
            warnings.warn("WARNING: empty image")

        color_data = color_data.squeeze()
        return color_data

    def get_depth(self) -> torch.Tensor:
        """
        Extract depth data from the camera data dictionary.

        Args:
            camera_data_dict (Dict): Dictionary containing camera data

        Returns:
            torch.Tensor: Depth image data

        Raises:
            AssertionError: If data format doesn't match expected shape or channels
        """
        depth_data = self._get_camera_output()["distance_to_image_plane"].clone()
        assert depth_data.shape[0] == 1, "We expect a single environment in Isaac Lab"
        assert depth_data.shape[-1] == 1, "We expect a single channel depth image"
        depth_data = depth_data.squeeze()
        return depth_data

    def get_semantic_segmentation(self) -> Tuple[torch.Tensor, Dict]:
        """
        Extract semantic segmentation data from the camera data dictionary.

        Returns:
            Tuple[torch.Tensor, Dict]: A tuple containing:
                - torch.Tensor: Semantic segmentation image data with shape (H, W, 4) and dtype uint8
                - Dict: Mapping from segmentation IDs (RGBA) to semantic labels (class names)
        """
        segmentation_data = self._get_camera_output()["semantic_segmentation"].clone()
        id_to_labels = self._camera.data.info[0]["semantic_segmentation"]["idToLabels"]
        assert segmentation_data.shape[0] == 1, "We expect a single environment in Isaac Lab"
        assert segmentation_data.shape[-1] == 4, "We expect a four channel segmentation image"
        segmentation_data = segmentation_data.squeeze()
        return segmentation_data.to(torch.uint8), id_to_labels

    def get_dynamic_segmentation(self, dynamic_class_labels: List[str]) -> torch.Tensor:
        """
        Extract the dynamic mask from the semantic segmentation data as a boolean mask.

        Args:
            dynamic_class_labels (List[str]): List of class labels to consider as dynamic objects.
                                            These labels must match the class names in the semantic segmentation.

        Returns:
            torch.Tensor: Boolean mask tensor of shape (H, W) where True indicates dynamic objects.
                         The mask is True for pixels belonging to any of the specified dynamic classes.
        """
        # Get the semantic segmentation data and the mapping from RGBDA IDs to labels.
        segmentation_data, rgba_to_labels = self.get_semantic_segmentation()

        # Select the RGBA values for the dynamic classes.
        dynamic_rgba_list = [
            eval(k)
            for k in rgba_to_labels.keys()
            if rgba_to_labels[k].get("class", None) in dynamic_class_labels
        ]

        # Create a boolean mask for pixels matching any of the selected dynamic RGBA IDs.
        dynamic_mask = torch.zeros_like(segmentation_data[..., 0], dtype=torch.bool)
        for dynamic_rgba in dynamic_rgba_list:
            assert len(dynamic_rgba) == 4, "RGBA ID must be a 4D tensor"
            # Compare the RGBA values of the segmentation data with the current RGBA ID.
            dynamic_mask |= (
                segmentation_data
                == torch.tensor(dynamic_rgba).to(segmentation_data.device).view(1, 1, 4)
            ).all(dim=-1)

        return dynamic_mask

    def get_pcd(self) -> torch.Tensor:
        """Get the pointcloud from the camera by backprojecting the depth image."""
        depth_frame = self.get_depth()
        intrinsics = self.get_intrinsics()
        translation, rotation_quat = self.get_pose()

        pointcloud = get_camera_pointcloud(intrinsics, depth_frame, translation, rotation_quat)

        return pointcloud

    def get_valid_depth_mask(self, min_depth: float = 0.0) -> torch.Tensor:
        """Get mask that indicates which depth values are aboice a theshold."""
        depth_frame = self.get_depth()
        valid_depth_mask = depth_frame > min_depth
        return valid_depth_mask

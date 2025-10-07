# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import os
from typing import List, Tuple

from PIL import Image
import numpy as np
import torch

from mindmap.common_utils.asserts import assert_in_bounds_of_type
from mindmap.embodiments.state_base import RobotStateBase
from mindmap.isaaclab_utils.isaaclab_camera_handler import IsaacLabCameraHandler
from mindmap.isaaclab_utils.isaaclab_datagen_utils import DemoOutcome
from mindmap.mapping.nvblox_mapper_constants import DEPTH_SCALE_FACTOR


class IsaacLabWriter:
    """
    A writer class for saving demonstration data from IsaacLab experiments.

    Args:
        output_dir (str): Base directory where all demonstration data will be saved
    """

    def __init__(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self._output_dir = output_dir

    @staticmethod
    def save_array_as_image(arr, output_path):
        """
        Convert and save a numpy array as an image file using PIL.

        Args:
            arr (np.ndarray): Array containing image data. Must be in a format
                compatible with PIL (e.g., uint8 for RGB, grayscale)
            output_path (str): Full path where the image will be saved

        Note:
            The array should be properly formatted (correct shape and data type)
            for the intended image format before calling this method.
        """
        im = Image.fromarray(arr)
        im.save(output_path)

    def write_depth_camera(self, camera_handler: IsaacLabCameraHandler, frame_index: int) -> None:
        """
        Write depth, intrinsics and pose from a depth camera to disk.

        Args:
            camera_handler (IsaacLabCameraHandler): Handler containing camera data including depth,
                intrinsics and pose information
            frame_index (int): Index of the current frame

        Note:
            Data is saved with consistent naming pattern: {frame_index:04}.{camera_name}_{type}.{ext}
            where type is one of: depth, intrinsics, or pose
        """
        camera_name = camera_handler.camera_name
        self.write_depth(camera_handler.get_depth(), camera_name, frame_index)
        self.write_intrinsics(camera_handler.get_intrinsics(), camera_name, frame_index)
        self.write_pose(camera_handler.get_pose(), camera_name, frame_index)

    def write_pose(
        self, pose: Tuple[torch.Tensor, torch.Tensor], camera_name: str, frame_index: int
    ) -> None:
        translation_W_C, rotation_W_C_quat = pose
        filename = f"{self._output_dir}/{frame_index:04}.{camera_name}_pose.npy"
        pose_array = torch.cat([translation_W_C, rotation_W_C_quat]).cpu().numpy()
        np.save(filename, pose_array)

    def write_rgb(self, rgb: torch.Tensor, camera_name: str, frame_index: int) -> None:
        """
        Write RGB image data to disk in both PNG and NPY formats.

        Args:
            rgb (torch.Tensor): RGB image data
            camera_name (str): Name of the camera
            frame_index (int): Index of the current frame
        """
        rgb_png_filepath = f"{self._output_dir}/{frame_index:04}.{camera_name}_rgb.png"
        self.save_array_as_image(rgb.cpu().numpy(), rgb_png_filepath)

    def write_depth(self, depth_data: torch.Tensor, camera_name: str, frame_index: int) -> None:
        """
        Write depth image data to disk in PNG and NPY formats, and save point cloud data.

        Args:
            depth_data (torch.Tensor): Depth image data
            camera_name (str): Name of the camera
            frame_index (int): Index of the current frame
        """
        depth_png_filepath = f"{self._output_dir}/{frame_index:04}.{camera_name}_depth.png"
        # Clamping the depth data to uint16 range (we have seen inf-values).
        depth_data = torch.clamp(
            depth_data, min=0.0, max=torch.iinfo(torch.uint16).max / DEPTH_SCALE_FACTOR - 1e-3
        )
        assert_in_bounds_of_type(depth_data * DEPTH_SCALE_FACTOR, torch.uint16)
        # Convert the depth data to a uint16 image.
        depth_data_uint16 = (depth_data * DEPTH_SCALE_FACTOR).to(torch.uint16).cpu().numpy()
        self.save_array_as_image(depth_data_uint16, depth_png_filepath)

    def write_intrinsics(
        self, intrinsics: torch.Tensor, camera_name: str, frame_index: int
    ) -> None:
        """
        Write camera intrinsic matrix to disk in numpy format.

        Args:
            intrinsics (torch.Tensor): Camera intrinsic matrix of shape (3, 3)
            camera_name (str): Name of the camera
            frame_index (int): Index of the current frame
        """
        intrinsics_filename = f"{self._output_dir}/{frame_index:04}.{camera_name}_intrinsics.npy"
        np.save(intrinsics_filename, intrinsics.cpu().numpy())

    def write_semantic(
        self, segmentation_data: torch.Tensor, camera_name: str, frame_index: int
    ) -> None:
        """
        Write semantic segmentation data to disk as a PNG image.

        Args:
            segmentation_data (torch.Tensor): Semantic segmentation data
            camera_name (str): Name of the camera
            frame_index (int): Index of the current frame
        """
        assert segmentation_data.dim() == 2
        filename = f"{self._output_dir}/{frame_index:04}.{camera_name}_semantic.png"
        self.save_array_as_image(segmentation_data.cpu().numpy(), filename)

    def write_state(self, state: RobotStateBase, frame_index: int):
        """
        Write the end effector state data to disk in numpy format.

        Args:
            state (RobotStateBase): The state of the robot to be writen to disk.
            frame_index (int): Index of the current frame

        Note:
            Data is saved as '{frame_index:04}.robot_state.npy' containing a concatenated array
            of all state components from all end effectors in order.
        """
        # State tensor
        state_tensor = state.to_tensor()

        # Save the combined state
        path = f"{self._output_dir}/{frame_index:04}.robot_state.npy"
        np.save(path, state_tensor.cpu().numpy())

    def write_outcome(self, outcome: DemoOutcome):
        """
        Write the demonstration outcome (success/failure) to a text file.

        Args:
            outcome (bool): Whether the demonstration was successful

        Note:
            The outcome is saved as an integer (1 for success, 0 for failure)
            in 'demo_successful.npy' in the demo path
        """
        filename = f"{self._output_dir}/demo_successful.npy"
        np.save(filename, np.array(int(outcome.value)))

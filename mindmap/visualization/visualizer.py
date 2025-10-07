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
import os
import threading
import time
from typing import Callable, Dict, Optional, Tuple

import PIL
import cv2
import numpy as np
import numpy.typing as npt
import nvblox_torch
from nvblox_torch.mapper import Mapper, QueryType
import open3d as o3d
import torch
from torchtyping import TensorType

from mindmap.cli.args import VisualizerArgs
from mindmap.data_loading.data_types import DataType, includes_mesh, includes_pcd, includes_rgb
from mindmap.embodiments.arm.controller import ARM_CLOSEDNESS_THRESHOLD
from mindmap.embodiments.humanoid.controller import HUMANOID_CLOSEDNESS_THRESHOLD
from mindmap.image_processing.feature_extraction import get_nvblox_feature_dim
from mindmap.image_processing.pca import apply_pca, apply_pca_return_projection
from mindmap.mapping.isaaclab_nvblox_mapper import IsaacLabNvbloxMapper
from mindmap.mapping.nvblox_mapper_constants import MAPPER_TO_ID, get_workspace_bounds
from mindmap.model_utils.normalization import unnormalize_pos
from mindmap.visualization.visualization import (
    visualize_arrow,
    visualize_axis,
    visualize_pointcloud,
    visualize_trajectory,
    visualize_voxel_grid,
)

# cv2.waitKey(1) leads to a black first frame.
# Increasing the time fixes the issue.
WAIT_KEY_TIME_MS = 200

# The index of the batch to visualize.
BATCH_IDX = 0


def key_in_sample(sample: dict, key: str) -> bool:
    """Check if a key exists in a sample."""
    return key in sample and sample[key] is not None


class Visualizer:
    """Visualizer for mindmap.

       A visualizer for visualizing a batch of data and (optionally) a prediction
       coming out of a model. This is intended to be used for debugging across the
       whole mindmap pipeline.

    Args:
        args: VisualizerArgs
    """

    def __init__(
        self,
        args: VisualizerArgs,
        pca_params: Optional[Tuple[TensorType, TensorType, TensorType]] = None,
    ):
        # Params
        self.args = args
        # The visualizers
        self.visualizers = {}
        self.visualizers["main"] = self._create_visualizer("Model Inputs and Prediction")
        self.visualizers["encoded_inputs"] = None  # Initialized when needed
        if self.args.visualize_attention_weights:
            self.visualizers["cross_attention"] = self._create_visualizer(
                "Gripper->context attention weights. White indicates points that are masked out."
            )
        self.view_controllers = {}
        # An event that is set when the space key is pressed
        self.key_pressed = threading.Event()
        self.pca_params = pca_params
        # An index to save images and pointclouds
        self.save_index = 0
        # We require this to know what size to cut the features to.
        self.embedding_dim = get_nvblox_feature_dim(args.feature_type)

    def visualize(
        self,
        sample: dict,
        data_type: DataType,
        prediction: Optional[torch.Tensor] = None,
        isaac_lab_nvblox_mapper: Optional[IsaacLabNvbloxMapper] = None,
    ):
        """Visualize a sample and (optionally) a prediction.

        Args:
            sample (dict): A sample from a dataset or closed loop.
            data_type (DataType): The type of input data being used.
            prediction (Optional[torch.Tensor], optional): A prediction from a model. Defaults to None.
            isaac_lab_nvblox_mapper (Optional[IsaacLabNvbloxMapper], optional): The nvblox mapper for integrating
                isaaclab images. If passed we will visualize the nvblox mesh and feature grid. Defaults to None.
        """
        for visualizer in self.visualizers.values():
            if visualizer is not None:
                visualizer.clear_geometries()
        self._initialize_view_controllers_if_needed(sample)
        # World axis
        visualize_axis(
            torch.tensor([0, 0, 0, 1, 0, 0, 0]), self.visualizers["main"], "WORLD", size=0.1
        )

        # Detect if we're in keypose mode.
        using_keyposes = key_in_sample(sample, "is_keypose")

        # Don't mark keypose frames when we are in keypose mode because they are the only ones we are
        # visualizing.
        highlight_keypose_image = False if using_keyposes else True

        # If we are, detect if this step is a keypose.
        is_keypose = (
            sample["is_keypose"][BATCH_IDX] if key_in_sample(sample, "is_keypose") else False
        )

        # Visualize the last nvblox integration images
        # Note(dtingdahl): This should be kept at the beginning of the visualization function to
        # ensure that PCA basis is computed from the dense images, rather than from a potentially
        # sparse pointcloud.
        if isaac_lab_nvblox_mapper is not None:
            self._visualize_nvblox_integration_images(
                isaac_lab_nvblox_mapper.last_nvblox_integration_images,
                is_keypose,
                highlight_keypose_image,
            )

        # Keyposes
        if using_keyposes:
            self._visualize_past_and_future_keyposes(sample)
        else:
            self._visualize_trajectories(sample)

        # RGB
        if includes_rgb(data_type):
            self._visualize_rgb_images(sample, is_keypose, highlight_keypose_image)
        # Pointclouds
        if includes_pcd(data_type):
            self._visualize_pointclouds(sample)
        # Mesh vertices
        if includes_mesh(data_type):
            self._visualize_mesh_vertices(sample)

        # Prediction
        if prediction is not None:
            if using_keyposes:
                self._visualize_keyposes(prediction, lambda x: f"PREDICTED i-{x}")
            else:
                self._visualize_trajectories(sample, prediction)

        # Visualize the current state of the map.
        if isaac_lab_nvblox_mapper is not None:
            mapper = isaac_lab_nvblox_mapper.mapper
            for mapper_id in range(mapper.num_mappers()):
                surface_points, surface_features = self._get_visual_surface_points_and_features(
                    mapper, mapper_id
                )
                if surface_points is not None and surface_features is not None:
                    mapper_name = MAPPER_TO_ID(mapper_id).name
                    self._visualize_nvblox_mesh(mapper, mapper_id, f"mesh_{mapper_name}")
                    self._visualize_nvblox_feature_grid(
                        mapper, mapper_id, f"feature_grid_{mapper_name}"
                    )

        # Encoded features
        if self.args.visualize_encoded_features:
            if "context" in sample and "context_feats" in sample:
                self._visualize_encoded_pointcloud(
                    sample["context"],
                    sample["context_feats"],
                )

        if self.args.visualize_attention_weights:
            if "context" in sample and "context_feats" in sample and "context_mask" in sample:
                self._visualize_attention_weights(
                    vertices=sample["context"],
                    weights=sample["cross_attn_weights"],
                    mask=sample["context_mask"],
                    visualizer_name="cross_attention",
                )

        # Set the viewpoint from the last sample
        for visualizer_name in self.view_controllers.keys():
            self.view_controllers[visualizer_name].restore_viewpoint(
                self.visualizers[visualizer_name]
            )

        # Save the image
        self._update_visualization(self.visualizers["main"])
        if self.args.visualizer_record_camera_output_path:
            self._save_composed_render_image(
                sample,
                out_dir=self.args.visualizer_record_camera_output_path,
                index=self.save_index,
                is_keypose=is_keypose,
            )
        self.save_index += 1

    def run_until_space_pressed(self):
        # Update rendering until key is pressed.
        # This makes it possible to modify the viewpoint with the mouse.
        while not self.key_pressed.is_set():
            for visualizer in self.visualizers.values():
                self._update_visualization(visualizer)
            time.sleep(0.01)
        self.key_pressed.clear()
        # Do one last update after the key is pressed
        for visualizer in self.visualizers.values():
            self._update_visualization(visualizer)
        # Save the viewpoint for the next sample
        for visualizer_name in self.view_controllers.keys():
            self.view_controllers[visualizer_name].store_camera_pose(
                self.visualizers[visualizer_name]
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("exiting")
        for visualizer in self.visualizers:
            visualizer.destroy_window()
        cv2.destroyAllWindows()

    def _update_visualization(self, visualizer: o3d.visualization.VisualizerWithKeyCallback):
        if visualizer is not None:
            visualizer.poll_events()
            visualizer.update_renderer()

    def _create_visualizer(self, window_name: str) -> o3d.visualization.VisualizerWithKeyCallback:
        visualizer = o3d.visualization.VisualizerWithKeyCallback()
        visualizer.create_window(width=800, height=600, window_name=window_name)
        visualizer.get_render_option().line_width = 50
        visualizer.get_render_option().point_size = self.args.visualizer_point_size
        visualizer.get_render_option().background_color = np.asarray(
            self.args.visualizer_background_rgb
        )
        visualizer.register_key_callback(ord(" "), lambda vis: self._on_key_press(vis))
        return visualizer

    def _on_key_press(self, vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
        # Set the flag to indicate that a key has been pressed
        self.key_pressed.set()
        # Return False to continue the event loop
        return False

    def _visualize_nvblox_integration_images(
        self,
        nvblox_integration_images: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        is_keypose: bool,
        highlight_keypose_image: bool,
    ):
        """Visualize a sample and (optionally) a prediction.

        Args:
            nvblox_integration_images (Optional[Dict[str, Dict[str, Dict[str, torch.Tensor]]]], optional):
                A nested dictionary of images used for nvblox integration to visualize:
                - First level key: Camera name (e.g. 'wrist_camera', 'pov_camera')
                - Second level key: Mapper name ('STATIC' or 'DYNAMIC')
                - Third level contains the integration images:
                    - depth_frame: Depth image used for integration
                    - depth_mask: Binary mask for valid depth values
                    - rgb_frame: RGB image used for integration
                    - rgb_mask: Binary mask for RGB integration
                    - feature_frame: Extracted features
                    - feature_mask: Binary mask for feature integration
                    - pcd: Point cloud used for feature extraction (only present in STATIC mapper)
            is_keypose (bool): Whether the sample is a keypose
            highlight_keypose_image (bool): Whether to highlight the keypose image
        """
        for camera_name in nvblox_integration_images.keys():
            for mapper_name in nvblox_integration_images[camera_name].keys():
                image_name_prefix = f"{camera_name}/{mapper_name}".lower()
                images_in_prefix = nvblox_integration_images[camera_name][mapper_name]

                if "rgb_frame" in images_in_prefix:
                    mask = images_in_prefix["rgb_mask"] if "rgb_mask" in images_in_prefix else None
                    self._visualize_rgb_image(
                        images_in_prefix["rgb_frame"],
                        f"{image_name_prefix}: rgb",
                        is_keypose,
                        highlight_keypose_image,
                        mask,
                    )

                # Feature images
                if "feature_frame" in images_in_prefix:
                    mask = (
                        images_in_prefix["feature_mask"]
                        if "feature_mask" in images_in_prefix
                        else None
                    )
                    self._visualize_feature_frame(
                        images_in_prefix["feature_frame"], f"{image_name_prefix}: feature", mask
                    )

                # Depth images
                if "depth_frame" in images_in_prefix:
                    mask = (
                        images_in_prefix["depth_mask"] if "depth_mask" in images_in_prefix else None
                    )
                    self._visualize_depth_image(
                        images_in_prefix["depth_frame"], f"{image_name_prefix}: depth", mask
                    )

                # Input images
                if "input_mask" in images_in_prefix:
                    mask = images_in_prefix["input_mask"]
                    self._visualize_mask(mask, f"{image_name_prefix}: input_mask")

                # Back projected features
                if self.args.visualize_backprojected_features:
                    if "feature_frame" in images_in_prefix and "pcd" in images_in_prefix:
                        self._visualize_feature_pcd(
                            images_in_prefix["feature_frame"],
                            images_in_prefix["pcd"],
                            f"{image_name_prefix}: pcd",
                        )

    def _visualize_keyposes(self, keyposes: torch.Tensor, idx_to_str: Callable[[int], str]):
        assert keyposes.shape[-1] >= 8
        # Add a gripper dimension if needed
        if keyposes.dim() == 3:
            keyposes = keyposes.unsqueeze(1)
        num_keyposes = keyposes.shape[1]

        for keypose_idx in range(num_keyposes):
            # We visualize each gripper's keypose independently
            num_grippers = keyposes.shape[2]
            for gripper_idx in range(num_grippers):
                pose = keyposes[BATCH_IDX][keypose_idx][gripper_idx][0:7]
                closedness = keyposes[BATCH_IDX][keypose_idx][gripper_idx][-1]
                closedness_threshold = (
                    ARM_CLOSEDNESS_THRESHOLD if num_grippers == 1 else HUMANOID_CLOSEDNESS_THRESHOLD
                )
                gripper_closed = closedness > closedness_threshold
                visualize_axis(
                    pose,
                    self.visualizers["main"],
                    idx_to_str(keypose_idx) + f" gripper{gripper_idx}",
                    size=0.02,
                    color=(1, 0, 0) if gripper_closed else (1, 1, 1),
                )

    def _visualize_depth_image(
        self, depth: torch.Tensor, name: str, mask: Optional[torch.Tensor] = None
    ):
        assert depth.dim() == 2
        depth_numpy = depth.cpu().numpy()
        depth_normalized = cv2.normalize(depth_numpy, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        if mask is not None:
            depth_colormap = self._mask_out_image(depth_colormap, mask.cpu().numpy())
        cv2.imshow(name, depth_colormap)
        cv2.waitKey(WAIT_KEY_TIME_MS)

    def _visualize_mask(self, mask: torch.Tensor, name: str):
        assert mask.dim() == 2
        cv2.imshow(name, (mask.cpu().numpy() * 255).astype(np.uint8))
        cv2.waitKey(WAIT_KEY_TIME_MS)

    def _visualize_rgb_image(
        self,
        rgb: torch.Tensor,
        name: str,
        is_keypose: bool = False,
        highlight_keypose_image: bool = True,
        mask: Optional[torch.Tensor] = None,
    ):
        assert rgb.dim() == 3
        assert rgb.shape[0] == 3
        rgb_numpy = (255 * rgb.permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
        if is_keypose and highlight_keypose_image:
            rgb_numpy[:, :, 0] = 255
        if mask is not None:
            rgb_numpy = self._mask_out_image(rgb_numpy, mask.cpu().numpy())
        cv2.imshow(name, cv2.cvtColor(rgb_numpy, cv2.COLOR_RGB2BGR))
        cv2.waitKey(WAIT_KEY_TIME_MS)

    def _visualize_rgb_images(self, sample: Dict, is_keypose: bool, highlight_keypose_image: bool):
        if key_in_sample(sample, "rgbs"):
            num_cams = sample["rgbs"].shape[1]
            assert num_cams <= 2, "We only support 1 or 2 cameras for now."
            for cam_idx in range(num_cams):
                self._visualize_rgb_image(
                    sample["rgbs"][BATCH_IDX][cam_idx],
                    f"RGB Cam{cam_idx} Image",
                    is_keypose,
                    highlight_keypose_image,
                )

    def _apply_pca(self, features: torch.Tensor) -> torch.Tensor:
        """Apply PCA to the features and return the PCA-projected features.

        Args:
            features (torch.Tensor): NxC features

        Returns:
            torch.Tensor: Nx3 PCA-projected features
        """
        assert features.dim() == 2
        if self.pca_params is None:
            features_pca, self.pca_params = apply_pca_return_projection(features.float())
            print(f"Computed PCA basis for {features.shape[0]} features")
        else:
            features_pca = apply_pca(features.float(), self.pca_params)
        assert features_pca.dim() == 2
        assert features_pca.shape[0] == features.shape[0]
        assert features_pca.shape[1] == 3
        return features_pca

    def _get_normalized_colors_from_features(self, features: torch.Tensor) -> torch.Tensor:
        # Remove the excess features caused by nvblox having a longer feature length than required.
        if features.shape[-1] > self.embedding_dim:
            features = features[..., : self.embedding_dim]
        features_pca = self._apply_pca(features)
        features_colors_normalized = cv2.normalize(
            features_pca.cpu().detach().numpy(),
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_64F,
        )
        return features_colors_normalized

    def _mask_out_image(self, image: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
        assert image.ndim == 3
        assert mask.ndim == 2
        assert mask.shape == image.shape[:2]
        mask_expanded = mask[..., np.newaxis]
        return image * mask_expanded

    def _visualize_feature_frame(
        self, feature_frame: torch.Tensor, visualizer_name: str, mask: Optional[torch.Tensor] = None
    ) -> None:
        """
        Visualize a CLIP feature frame by projecting features to RGB colors using PCA.

        Args:
            feature_frame (torch.Tensor): Feature tensor of shape (H, W, C) containing CLIP features
            visualizer_name (str): Name of the OpenCV window to display the visualization
            mask (torch.Tensor): Mask of shape (H, W) to apply to the features
        """
        assert feature_frame.dim() == 3
        frame_shape = feature_frame.shape
        flattened_features = feature_frame.reshape(-1, feature_frame.shape[-1])
        features_colors = self._get_normalized_colors_from_features(flattened_features)
        # Show the PCA-projected features as an RGB image using OpenCV
        features_colors_normalized_rgb = (features_colors * 255).astype(np.uint8)
        features_colors_normalized_rgb = features_colors_normalized_rgb.reshape(
            frame_shape[0], frame_shape[1], 3
        )
        # Mask out the image
        if mask is not None:
            features_colors_normalized_rgb = self._mask_out_image(
                features_colors_normalized_rgb, mask.cpu().numpy()
            )
        cv2.imshow(visualizer_name, cv2.cvtColor(features_colors_normalized_rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(WAIT_KEY_TIME_MS)

    def _visualize_past_and_future_keyposes(self, sample: Dict):
        # Visualize past keyposes.
        hist_length = sample["gripper_history"].shape[1]
        self._visualize_keyposes(
            sample["gripper_history"], lambda x: f"KEYPOSE i-{hist_length - x}"
        )
        # Visualize next keypose (if the sample has GT, i.e. in training/open-loop)
        if key_in_sample(sample, "gt_gripper_pred"):
            self._visualize_keyposes(sample["gt_gripper_pred"], lambda x: f"KEYPOSE i+{x}")

        # Visualize the head yaw history.
        if key_in_sample(sample, "gt_head_yaw"):
            self._visualize_head_yaw(sample["gt_head_yaw"], lambda x: f"KEYPOSE i+{x}")

    def _visualize_head_yaw(self, head_yaw: torch.Tensor, idx_to_str: Callable[[int], str]):
        assert head_yaw.dim() == 3
        assert head_yaw.shape[2] == 1

        num_keyposes = head_yaw.shape[1]
        for keypose_idx in range(num_keyposes):
            head_yaw_value = head_yaw[BATCH_IDX][keypose_idx][keypose_idx].item()

            # Compute the direction vector from yaw
            direction = np.array([np.cos(head_yaw_value), np.sin(head_yaw_value), 0.0])
            visualize_arrow(
                direction=direction,
                origin=np.array([0, 0, 0.3]),
                visualizer=self.visualizers["main"],
                text=idx_to_str(keypose_idx) + f" {int(head_yaw_value / np.pi * 180)} deg",
            )

    def _visualize_trajectories(self, sample: Dict):
        # Visualize the gripper history.
        if key_in_sample(sample, "gripper_history"):
            visualize_trajectory(
                sample["gripper_history"][BATCH_IDX], self.visualizers["main"], color=[1, 0, 0]
            )  # red
        # Visualize the ground truth trajectory (if the sample has GT, i.e. in training/open-loop)
        if key_in_sample(sample, "gt_gripper_pred"):
            visualize_trajectory(
                sample["gt_gripper_pred"][BATCH_IDX], self.visualizers["main"], color=[0, 1, 0]
            )  # green

    def _visualize_encoded_pointcloud(self, context, context_feats):
        # Unnormalize the pointcloud for visualization.
        # NOTE: The pointcloud is normalized to the workspace bounds inside the encoder,
        # along with the trajectory. We unnormalize it here for visualization.
        # context = unnormalize_pointcloud(context, sample['workspace_bounds'])
        context = unnormalize_pos(context, get_workspace_bounds(self.args.task))
        # We're only visualizing the first batch but let's use all of them to compute the
        # pca basis.
        dim = context_feats.shape[-1]
        _, pca_basis = apply_pca_return_projection(context_feats.view(-1, dim))
        feats_pca = apply_pca(context_feats[0], pca_basis)
        if self.visualizers["encoded_inputs"] is None:
            self.visualizers["encoded_inputs"] = self._create_visualizer("Encoded Model Inputs")
        pcd = visualize_pointcloud(
            context[0].cpu().detach().numpy(),
            self.visualizers["encoded_inputs"],
            255 * feats_pca.cpu().detach().numpy(),
            compute_normals=False,
            max_distance=self.args.visualizer_pointcloud_max_distance,
        )

        self._maybe_write_pcd_to_output_dir(pcd, f"encoded_inputs_{self.save_index:0>5}.ply")

    def _maybe_write_pcd_to_output_dir(self, pcd: o3d.geometry.PointCloud, filename: str):
        if self.args.visualizer_pointclouds_ply_output_dir is not None:
            path = os.path.join(self.args.visualizer_pointclouds_ply_output_dir, filename)
            o3d.io.write_point_cloud(path, pcd)
            print(f"Saved pointcloud to {path}")

    def _visualize_attention_weights(
        self,
        vertices: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
        visualizer_name: str,
    ):
        # Visualize first batch
        mask = mask[0].cpu().numpy()
        weights = weights[0].detach().cpu().numpy().squeeze()
        vertices = unnormalize_pos(vertices, get_workspace_bounds(self.args.task))
        vertices = vertices[0].detach().cpu().numpy()

        # Normalize weights using masked points only
        active_weights = weights[mask]
        normalized_weights = (weights - active_weights.min()) / (
            active_weights.max() - active_weights.min()
        )

        weights_bgr = cv2.applyColorMap(
            (255 * normalized_weights).astype(np.uint8), cv2.COLORMAP_JET
        )
        weights_rgb = cv2.cvtColor(weights_bgr, cv2.COLOR_BGR2RGB).squeeze(1)
        weights_rgb[~mask] = 255  # highlight inactive points

        # Remove ones below threshold
        above_threshold_mask = normalized_weights > self.args.visualizer_min_attention_weight
        weights_rgb = weights_rgb[above_threshold_mask]
        vertices = vertices[above_threshold_mask]

        pcd = visualize_pointcloud(
            vertices,
            self.visualizers[visualizer_name],
            weights_rgb,
            compute_normals=False,
        )

        self._maybe_write_pcd_to_output_dir(pcd, f"attention_weights_{self.save_index:0>5}.ply")

    def _visualize_pointclouds(self, sample: Dict):
        # Show the pointclouds of the scene.
        if key_in_sample(sample, "pcds"):
            num_cams = sample["pcds"].shape[1]
            assert num_cams <= 2, "We only support 1 or 2 cameras for now."
            pcd_vec = []
            rgb_vec = []
            for cam_idx in range(num_cams):
                pcd_vec.append(
                    sample["pcds"][BATCH_IDX][cam_idx].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
                )
                rgb_vec.append(
                    sample["rgbs"][BATCH_IDX][cam_idx].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
                )
                # Slight red tint to the wrist pointcloud so we can distinguish it.
                if cam_idx == 1:
                    rgb_vec[-1] = rgb_vec[-1] * np.array([1, 0.75, 0.75])
            pcd_all = np.concatenate(pcd_vec)
            rgb_all = 255 * np.concatenate(rgb_vec)
            visualize_pointcloud(
                pcd_all,
                self.visualizers["main"],
                rgb_all,
                max_distance=self.args.visualizer_pointcloud_max_distance,
            )

    def _visualize_feature_pcd(
        self, feature_frame: torch.Tensor, feature_pcd: torch.Tensor, name: str
    ) -> None:
        """
        Visualize a feature pointcloud by projecting features to RGB colors using PCA.

        Args:
            feature_frame (torch.Tensor): Feature tensor of shape (H, W, C) containing CLIP features
            feature_pcd (torch.Tensor): Pointcloud tensor of shape (H, W, 3) containing pointcloud
            name (str): Name of the Open3d visualizer to use.
        """
        if name not in self.visualizers:
            self.visualizers[name] = self._create_visualizer(name)
        self.visualizers[name].clear_geometries()
        features_flat = feature_frame.reshape(-1, feature_frame.shape[-1])
        colors_flat = self._get_normalized_colors_from_features(features_flat) * 255.0
        pointcloud_flat = feature_pcd.reshape(-1, 3)
        pcd = visualize_pointcloud(
            pointcloud_flat.cpu().numpy(),
            self.visualizers[name],
            colors_flat,
            max_distance=self.args.visualizer_pointcloud_max_distance,
            compute_normals=False,
        )
        self.visualizers[name].update_renderer()

        self._maybe_write_pcd_to_output_dir(pcd, f"feature_pcd_{self.save_index:0>5}.ply")

    def _visualize_mesh_vertices(self, sample: dict):
        if key_in_sample(sample, "vertices") and key_in_sample(sample, "vertex_features"):
            # Visualize the featurized mesh vertices
            vertices = sample["vertices"].squeeze().cpu().numpy()
            vertex_colors = self._apply_pca(sample["vertex_features"].squeeze())
            assert isinstance(vertices, np.ndarray)
            assert isinstance(vertex_colors, torch.Tensor)
            pcd = visualize_pointcloud(
                vertices,
                self.visualizers["main"],
                (vertex_colors * 255.0).to(torch.uint8).cpu().numpy(),
            )
            self._maybe_write_pcd_to_output_dir(
                pcd, f"mesh_vertex_features_{self.save_index:0>5}.ply"
            )

    def _visualize_nvblox_mesh(self, mapper: Mapper, mapper_id: int, name: str) -> None:
        mapper.update_color_mesh(mapper_id)
        mesh = mapper.get_color_mesh(mapper_id)
        if mesh is None:
            return
        if name not in self.visualizers:
            self.visualizers[name] = self._create_visualizer(name)
        self.visualizers[name].clear_geometries()
        self.visualizers[name].add_geometry(mesh.to_open3d())
        self.visualizers[name].update_renderer()

        if self.args.visualizer_pointclouds_ply_output_dir is not None:
            path = os.path.join(
                self.args.visualizer_pointclouds_ply_output_dir,
                f"nvblox_mesh_{self.save_index:0>5}.ply",
            )
            mesh.save(path)
            print(f"Saved mesh to {path}")

    def _get_visual_surface_points_and_features(
        self, mapper: Mapper, mapper_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tsdf_layer = mapper.tsdf_layer_view(mapper_id=mapper_id)
        # Look for the surface voxels using the TSDF.
        tsdf_and_weights, points = tsdf_layer.get_tsdfs_below_zero()
        tsdf_weights = tsdf_and_weights[:, 1]
        tsdf_weights_mask = tsdf_weights.cpu() > self.args.visualizer_min_tsdf_weight
        if points.shape[0] == 0:
            return None, None

        # Query the features and weights for the surface voxels.
        features_and_weight = mapper.query_layer(
            query_type=QueryType.FEATURE, query=points, mapper_id=mapper_id
        )
        features = features_and_weight[:, :-1]
        points = points[tsdf_weights_mask]
        features = features[tsdf_weights_mask]

        return points, features

    def _get_feature_surface_points_and_colors(
        self, mapper: Mapper, mapper_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve surface 3D points and corresponding feature colors for visualization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - points (Tensor): The 3D coordinates of surface voxels.
                - feature_colors (Tensor): The colors (in [0,1]) derived via PCA of features.
        """
        points, features = self._get_visual_surface_points_and_features(mapper, mapper_id)
        features_colors = self._get_normalized_colors_from_features(features) * 255.0
        return points, features_colors

    def _visualize_nvblox_feature_grid(self, mapper: Mapper, mapper_id: int, name: str) -> None:
        """
        Visualize the current NVblox 3D feature grid as a pointcloud with PCA-projected colors.
        """
        if name not in self.visualizers:
            self.visualizers[name] = self._create_visualizer(name)
        self.visualizers[name].clear_geometries()
        # Get the surface voxels for visualization and the correspoinding PCA projection of the features.
        feature_points, feature_colors = self._get_feature_surface_points_and_colors(
            mapper, mapper_id=mapper_id
        )
        if feature_points.shape[0] == 0 or feature_colors.shape[0] == 0:
            print("No features to visualize")
            return
        visualize_voxel_grid(
            feature_points.cpu(),
            self.visualizers[name],
            feature_colors,
            voxel_size=self.args.visualizer_voxel_size_m,
        )
        self.visualizers[name].update_renderer()

    def _initialize_view_controllers_if_needed(self, sample: dict):
        for visualizer_name in self.visualizers.keys():
            if visualizer_name not in self.view_controllers:
                if key_in_sample(sample, "gripper_history"):
                    # We can look at the first gripper for now.
                    lookat = sample["gripper_history"][0, 0, 0, :3].cpu().numpy() + np.array(
                        [0, 0, -0.18]
                    )
                else:
                    lookat = (0, 0, 0)
                self.view_controllers[visualizer_name] = ViewPointController(lookat=lookat)

    def _save_composed_render_image(
        self,
        sample,
        out_dir: str,
        index: int,
        is_keypose: bool = False,
    ):
        """Save image of the 3D view with cameras overlayed"""

        # Render 3D view to array
        render = np.asarray(self.visualizers["main"].capture_screen_float_buffer(do_render=True))

        # Create PIL images
        background_image = PIL.Image.fromarray(np.uint8(render * 255))
        font = PIL.ImageFont.truetype("DejaVuSans.ttf", 16)

        # Add the RGB images to image of the 3D scene.
        num_cams = sample["rgbs"].shape[1]
        assert num_cams <= 2, "We only support 1 or 2 cameras for now."
        for cam_idx in range(num_cams):
            rgb_image = PIL.Image.fromarray(
                np.uint8(255 * sample["rgbs"][BATCH_IDX][cam_idx].permute(1, 2, 0).numpy())
            )
            PIL.ImageDraw.Draw(rgb_image).text((0, 0), "RGB", (255, 255, 255), font=font)
            rgb_image = PIL.ImageOps.expand(rgb_image, border=5, fill="white")
            position_rgb = (
                background_image.width - rgb_image.width,
                background_image.height - (cam_idx + 1) * rgb_image.height - cam_idx * 10,
            )
            background_image.paste(rgb_image, position_rgb)

        # Highlight keyposes by adding a color tint and a text label
        if is_keypose:
            pixels = np.array(background_image)
            pixels[:, :, 0] = 0
            background_image = PIL.Image.fromarray(pixels)
            kp_font = PIL.ImageFont.truetype("DejaVuSans.ttf", 32)
            PIL.ImageDraw.Draw(background_image).text(
                (-1, background_image.width / 2), "!!KEYPOSE!!", (255, 255, 255), font=kp_font
            )

        # Save to disk
        fname = f"renderer_compose.{index:0>5}.png"
        out_path = os.path.join(out_dir, fname)
        background_image.save(out_path)


@dataclass
class ViewPointController:
    """Stores viewpoint info.

    O3d resets the view every time new geometry is added. This class can restore a
    view obtained from manually moving the camera.
    """

    lookat: npt.NDArray
    up: npt.NDArray = np.array([0, 0, 0.5])
    front: npt.NDArray = np.array([1, 0.1, 0.4])
    zoom: float = 0.3
    _camera_params = None

    def store_camera_pose(self, visualizer: o3d.visualization.Visualizer):
        """Store the current viewpoint. Call this after view has been adjusted by user."""
        if visualizer is not None:
            view_control = visualizer.get_view_control()
            self._camera_params = view_control.convert_to_pinhole_camera_parameters()

    def restore_viewpoint(self, visualizer: o3d.visualization.Visualizer):
        """Restores viewpoint from the stored one and sets view direction. Call this after new geometry has been added"""
        if visualizer is not None:
            view_control = visualizer.get_view_control()
            view_control.set_lookat(self.lookat)
            view_control.set_up(self.up)
            view_control.set_front(self.front)
            view_control.set_zoom(self.zoom)
            view_control.camera_local_translate(0, 0, 0.25)

            # If camera_params has been set, all the above will be overridden with the values stored in _camera_params
            if self._camera_params:
                view_control.convert_from_pinhole_camera_parameters(self._camera_params, True)

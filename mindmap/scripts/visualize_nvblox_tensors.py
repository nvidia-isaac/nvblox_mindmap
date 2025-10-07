#!/usr/bin/env python
#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import argparse
import pathlib
from typing import Any

import open3d as o3d

from mindmap.data_loading.batching import unpack_batch
from mindmap.data_loading.data_types import DataType
from mindmap.data_loading.dataset import SamplingWeightingType
from mindmap.data_loading.dataset_files_by_encoding_method import (
    get_data_loader_without_augmentations,
)
from mindmap.data_loading.vertex_sampling import VertexSamplingMethod
from mindmap.embodiments.arm.embodiment import ArmEmbodiment
from mindmap.embodiments.embodiment_base import EmbodimentBase, EmbodimentType
from mindmap.embodiments.humanoid.embodiment import HumanoidEmbodiment
from mindmap.image_processing.feature_extraction import FeatureExtractorType
from mindmap.image_processing.pca import apply_pca_return_projection
from mindmap.keyposes.keypose_detection_mode import KeyposeDetectionMode
from mindmap.tasks.tasks import Tasks

VOXEL_SIZE_M = 0.01
DISTANCE_LIMIT = 0.01


def get_sample(
    embodiment: EmbodimentBase,
    demos: str,
    demo_path: pathlib.Path,
    idx: int,
    data_type: DataType,
) -> Any:
    data_loader, _ = get_data_loader_without_augmentations(
        embodiment=embodiment,
        dataset_path=str(demo_path),
        demos=demos,
        num_workers=0,
        batch_size=1,
        use_keyposes=True,
        data_type=data_type,
        extra_keyposes_around_grasp_events=[5],
        keypose_detection_mode=KeyposeDetectionMode.HIGHEST_Z_BETWEEN_GRASP,
        include_failed_demos=False,
        gripper_encoding_mode="binary",
        num_history=3,
        prediction_horizon=1,
        add_external_cam=True,
        num_vertices_to_sample=None,
        vertex_sampling_method=VertexSamplingMethod.NONE,
        sampling_weighting_type=SamplingWeightingType.NONE,
        seed=0,
    )

    # Iterate through to the requested batch
    assert idx < len(data_loader)
    for i, batch in enumerate(data_loader):
        if i == idx:
            break
    batch = unpack_batch(
        embodiment=embodiment,
        batch=batch,
        batch_size=1,
        data_type=data_type,
        feature_type=FeatureExtractorType.RGB,
        image_size=(256, 256),
        num_history=3,
        add_external_cam=True,
        device="cuda",
    )
    return batch


class Visualizers:
    def __init__(self):
        self.visualizers = []

    def add_visualizer(self, name: str, geometry: o3d.geometry.Geometry):
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(name)
        visualizer.add_geometry(geometry)
        self.visualizers.append(visualizer)

    def visualize(self):
        while True:
            for visualizer in self.visualizers:
                visualizer.poll_events()
                visualizer.update_renderer()


def visualize_nvblox_sample(
    embodiment_type: EmbodimentType,
    task: Tasks,
    demo_path: pathlib.Path,
    idx: int,
    data_type: DataType,
):
    assert demo_path.exists(), f"Demo path {demo_path} does not exist"
    visualizers = Visualizers()

    if embodiment_type == EmbodimentType.ARM:
        embodiment = ArmEmbodiment()
    elif embodiment_type == EmbodimentType.HUMANOID:
        embodiment = HumanoidEmbodiment(task)
    else:
        raise ValueError(f"Embodiment type {embodiment_type} not supported")

    # Get a sample
    # NOTE(alexmillane): The "0" here means we're looking at the first sample
    #                    in the batch.
    batch = get_sample(
        embodiment,
        demo_path,
        idx=idx,
        data_type=data_type,
    )

    if data_type == DataType.MESH:
        vertices = batch["vertices"][0]
        vertex_features = batch["vertex_features"][0]

        print("Visualizing Vertex Feature Point Cloud")
        vertice_colors, _ = apply_pca_return_projection(vertex_features)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(vertice_colors.squeeze().cpu().numpy())
        visualizers.add_visualizer("Vertex Feature Point Cloud", pcd)

    else:
        raise ValueError(f"Data type {data_type} not supported")

    visualizers.visualize()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Visualize a sample from the voxel data")
    parser.add_argument(
        "--demo_dir", type=str, required=True, help="The directory containing a demo's data."
    )
    parser.add_argument("--data_type", type=str, required=True, help="The feature grid data type.")
    parser.add_argument(
        "--idx", type=int, required=True, help="The idx of the sample to visualize."
    )
    parser.add_argument("--embodiment_type", type=str, default="arm", help="The embodiment type.")
    args = parser.parse_args()

    # Visualize a sample
    visualize_nvblox_sample(
        args.embodiment_type,
        args.task,
        pathlib.Path(args.demo_dir),
        args.idx,
        DataType(args.data_type),
    )

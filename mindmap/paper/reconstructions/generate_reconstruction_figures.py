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
from typing import Tuple

import imageio.v3 as iio
import numpy as np
from nvblox_torch.mapper import Mapper
import open3d as o3d
import torch

from mindmap.paper.utils.utils import draw_geometries, get_open3d_feature_cubes_mesh

REGENERATE_VIEWPOINT = False

TASK_TO_IDX = {
    "cube_stacking": 20,
    "mug_in_drawer": 20,
    "drill_in_box": 50,
    "stick_in_bin": 100,
}


TrimBox = Tuple[int, int, int, int]


def get_trim_box(image: np.ndarray) -> TrimBox:
    bg_cols = np.all(np.all(image == 255, axis=-1), axis=0)
    bg_rows = np.all(np.all(image == 255, axis=-1), axis=1)
    first_col = np.where(bg_cols == False)[0][0]
    last_col = np.where(bg_cols == False)[0][-1]
    first_row = np.where(bg_rows == False)[0][0]
    last_row = np.where(bg_rows == False)[0][-1]
    return first_row, last_row, first_col, last_col


def get_minimal_trim_box(box_1: TrimBox, box_2: TrimBox) -> TrimBox:
    return (
        min(box_1[0], box_2[0]),
        max(box_1[1], box_2[1]),
        min(box_1[2], box_2[2]),
        max(box_1[3], box_2[3]),
    )


def trim_image(image: np.ndarray, box: TrimBox) -> np.ndarray:
    return image[box[0] : box[1], box[2] : box[3], :]


def trim_and_resave(color_path: pathlib.Path, feature_path: pathlib.Path):
    # Load the images
    color_image = iio.imread(color_path)
    color_image = np.asarray(color_image)
    feature_image = iio.imread(feature_path)
    feature_image = np.asarray(feature_image)

    # Get the minimal trim box
    color_box = get_trim_box(color_image)
    feature_box = get_trim_box(feature_image)
    minimal_box = get_minimal_trim_box(color_box, feature_box)

    # Trim the images
    color_image = trim_image(color_image, minimal_box)
    feature_image = trim_image(feature_image, minimal_box)

    # Save the trimmed images
    iio.imwrite(color_path, color_image)
    iio.imwrite(feature_path, feature_image)


def generate_reconstruction_figures(task_name: str, recompute_pca: bool = False):
    if task_name not in TASK_TO_IDX:
        raise ValueError(f"Unknown task: {task_name}. Must be one of {list(TASK_TO_IDX.keys())}")

    # Paths
    IDX = TASK_TO_IDX[task_name]
    script_dir = pathlib.Path(__file__).parent
    input_dir = script_dir / "input" / task_name
    output_dir = script_dir / "output" / task_name
    map_path = input_dir / f"{IDX:04d}.nvblox_map_static.nvblx"
    pca_params_path = input_dir / "pca_params.pt"
    viewpoint_path = input_dir / "viewpoint.json"

    # Create paths if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the map
    mapper = Mapper(voxel_sizes_m=0.01)
    mapper.load_from_file(str(map_path), 0)

    # Color Mesh
    mapper.update_color_mesh()
    color_mesh = mapper.get_color_mesh(0)
    color_mesh_o3d = color_mesh.to_open3d()

    # Feature Mesh
    mapper.update_feature_mesh()
    # Get the voxel mesh and PCA specification.
    if recompute_pca or not pca_params_path.exists():
        feature_mesh, pca_specification = get_open3d_feature_cubes_mesh(mapper)
        torch.save(pca_specification, pca_params_path)
    else:
        pca_specification = torch.load(pca_params_path)
        feature_mesh, pca_specification = get_open3d_feature_cubes_mesh(mapper, pca_specification)

    # Regenerate viewpoint if required
    if REGENERATE_VIEWPOINT or not viewpoint_path.exists():
        print(f"Please capture the viewpoint and save to: {viewpoint_path}")
        o3d.visualization.draw_geometries([feature_mesh])

    feature_output_path = output_dir / f"{task_name}_feature_cubes_mesh.png"
    draw_geometries([feature_mesh], viewpoint_path=viewpoint_path, output_path=feature_output_path)

    color_output_path = output_dir / f"{task_name}_color_mesh.png"
    draw_geometries([color_mesh_o3d], viewpoint_path=viewpoint_path, output_path=color_output_path)

    # Trim the images
    trim_and_resave(color_output_path, feature_output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate reconstruction figures for a given task")
    parser.add_argument(
        "--task_name",
        type=str,
        choices=list(TASK_TO_IDX.keys()),
        help="Name of the task to generate figures for",
    )
    parser.add_argument(
        "--recompute_pca",
        action="store_true",
        help="Whether to recompute the PCA parameters",
    )
    args = parser.parse_args()

    generate_reconstruction_figures(args.task_name, args.recompute_pca)


if __name__ == "__main__":
    main()

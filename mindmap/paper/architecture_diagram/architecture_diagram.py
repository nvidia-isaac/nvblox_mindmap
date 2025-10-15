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
from copy import deepcopy
import pathlib
from typing import Tuple

import cv2
import einops
import imageio.v3 as iio
from matplotlib import pyplot as plt
import numpy as np
from nvblox_torch.mapper import Mapper
import open3d as o3d
import torch

from mindmap.image_processing.backprojection import get_camera_pointcloud
from mindmap.image_processing.feature_extraction import RadioV25BFeatureExtractor
from mindmap.paper.utils.utils import (
    PCASpecification,
    colors_from_features,
    draw_geometries,
    get_open3d_feature_cubes_mesh,
)

SCRIPT_DIR = pathlib.Path(__file__).parent

# Size of the spheres for the pointcloud visualization
SPHERE_RADIUS = 0.0025

# Colors for the RGBD_ + Reconstruction overlay
RGBD_COLOR = np.array([236, 212, 68]) / 255.0
RECONSTRUCTION_COLOR = np.array([195, 47, 252]) / 255.0
RGBD_DELTA_POSITION = np.array([0.00, 0.0, 0.01])


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate architecture diagrams for MindMap paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing the input data files",
    )
    parser.add_argument(
        "--idx", type=int, default=20, help="Index for the data files (e.g., 20 for 0020.* files)"
    )
    parser.add_argument(
        "--recompute_pca",
        action="store_true",
        help="Recompute PCA parameters instead of loading from file",
    )

    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=SCRIPT_DIR / "output",
        help="Directory to save output images",
    )

    return parser.parse_args()


def load_voxel_mesh_and_pca(
    map_path: pathlib.Path,
    recompute_pca: bool = False,
    pca_params_path: pathlib.Path | None = None,
) -> Tuple[o3d.geometry.TriangleMesh, PCASpecification]:
    # Load the nvblox map.
    mapper = Mapper(voxel_sizes_m=0.01)
    mapper.load_from_file(str(map_path), 0)

    # Get the voxel mesh and PCA specification.
    if recompute_pca:
        voxel_mesh, pca_specification = get_open3d_feature_cubes_mesh(mapper)
        torch.save(pca_specification, pca_params_path)
    else:
        pca_specification = torch.load(pca_params_path)
        voxel_mesh, pca_specification = get_open3d_feature_cubes_mesh(mapper, pca_specification)

    return voxel_mesh, pca_specification


def show_and_save_voxel_mesh(
    voxel_mesh: o3d.geometry.TriangleMesh,
    output_path: pathlib.Path,
    viewpoint_path: pathlib.Path | None = None,
):
    # Visualize the voxel mesh
    draw_geometries([voxel_mesh], output_path=output_path, viewpoint_path=viewpoint_path)


def set_background_to_white(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        background_flag = np.all(image == 0, axis=-1)
    else:
        background_flag = image == 0
    if image.dtype == np.uint8:
        image[background_flag] = 255
    else:
        image[background_flag] = 1.0
    return image


def shear_image(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    rows, cols = image.shape[:2]
    # Shear parameters
    shear_x = 0.0  # tangent of shear angle
    shear_y = 0.5  # vertical shear
    M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    warped = cv2.warpAffine(image, M, (cols + int(rows * shear_x), rows + int(cols * shear_y)))
    # warped = set_background_to_white(warped)
    return warped


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    return image


def float_to_rgb(image: np.ndarray) -> np.ndarray:
    from matplotlib import colormaps

    cmap = colormaps["viridis"]
    background_flag = image == 0
    # image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    min_depth = np.min(image[~background_flag])
    max_depth = np.max(image[~background_flag])
    image_normalized = (image - min_depth) / (max_depth - min_depth)
    color = cmap(image_normalized)[:, :, :3]
    color[background_flag] = 1.0
    return color


def load_images_warp_show_and_save(
    depth_image_path: pathlib.Path,
    color_image_path: pathlib.Path,
    output_dir: pathlib.Path,
    pca_specification: PCASpecification,
):
    # Read the images
    depth_image = iio.imread(depth_image_path)
    color_image = iio.imread(color_image_path)

    # Extract features
    color_tensor = torch.from_numpy(color_image).to(device="cuda")
    color_tensor = einops.rearrange(color_tensor, "h w c -> 1 h w c")
    feature_extractor = RadioV25BFeatureExtractor(
        feature_image_size=(32, 32),
        desired_output_size=(512, 512),
    )
    features = feature_extractor.compute(color_tensor)
    features_flat = einops.rearrange(features, "b h w c -> (b h w) c")
    features_pca_flat, _ = colors_from_features(features_flat, pca_specification)
    features_pca = einops.rearrange(features_pca_flat, "(h w) c -> h w c", h=512, w=512)

    # Plot
    plt.subplot(2, 2, 1)
    plt.imshow(depth_image)
    plt.subplot(2, 2, 2)
    plt.imshow(color_image)
    plt.subplot(2, 2, 3)
    plt.imshow(features_pca.cpu().numpy())
    plt.show()

    # Warp
    warped_color_image = set_background_to_white(shear_image(color_image))
    warped_depth_image = float_to_uint8(float_to_rgb(shear_image(depth_image)))
    warped_features_pca = set_background_to_white(float_to_uint8(shear_image(features_pca)))

    # Save the images
    cv2.imwrite(
        output_dir / "warped_color_image.png", cv2.cvtColor(warped_color_image, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        output_dir / "warped_depth_image.png", cv2.cvtColor(warped_depth_image, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        output_dir / "warped_features_pca.png", cv2.cvtColor(warped_features_pca, cv2.COLOR_RGB2BGR)
    )

    # Plot
    plt.subplot(2, 2, 1)
    plt.imshow(warped_color_image)
    plt.subplot(2, 2, 2)
    plt.imshow(warped_depth_image)
    plt.subplot(2, 2, 3)
    plt.imshow(warped_features_pca)
    plt.show()

    return depth_image, features_pca


def show_and_save_depth_pointcloud(
    depth_image: np.ndarray,
    features_pca: np.ndarray,
    intrinsics_path: pathlib.Path,
    pose_path: pathlib.Path,
    output_dir: pathlib.Path,
    viewpoint_path: pathlib.Path | None = None,
) -> list[o3d.geometry.TriangleMesh]:
    # Load
    intrinsics = np.load(intrinsics_path)
    pose = np.load(pose_path)

    metric_depth_image = depth_image.astype(np.float32) / 1000.0
    backprojected_pointcloud = (
        get_camera_pointcloud(
            intrinsics=torch.from_numpy(intrinsics).to(device="cuda").unsqueeze(0),
            depth=torch.from_numpy(metric_depth_image).to(device="cuda").unsqueeze(0),
            position=torch.from_numpy(pose[:3]).to(device="cuda").unsqueeze(0),
            orientation=torch.from_numpy(pose[3:]).to(device="cuda").unsqueeze(0),
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )

    # Subsample to get the original feature pointcloud size
    backprojected_pointcloud_sub = backprojected_pointcloud[::16, ::16, :]
    point_colors_sub = features_pca[::16, ::16, :]

    # Flatten
    backprojected_pointcloud_sub = einops.rearrange(
        backprojected_pointcloud_sub, "h w c -> (h w) c"
    )
    point_colors_sub = einops.rearrange(point_colors_sub, "h w c -> (h w) c")

    # Make spheres
    sphere_prototype = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
    sphere_prototype.compute_vertex_normals()
    spheres = []
    for i in range(backprojected_pointcloud_sub.shape[0]):
        sphere = deepcopy(sphere_prototype)
        sphere.translate(backprojected_pointcloud_sub[i, :].cpu().numpy())
        sphere.paint_uniform_color(point_colors_sub[i, :].cpu().numpy())
        spheres.append(sphere)

    draw_geometries(spheres, output_path=output_dir / "rgbd.png", viewpoint_path=viewpoint_path)

    return spheres


def save_rgbd_plus_voxel_fig(
    rgbd_spheres_list: list[o3d.geometry.TriangleMesh],
    voxel_mesh: o3d.geometry.TriangleMesh,
    output_dir: pathlib.Path,
    viewpoint_path: pathlib.Path | None = None,
) -> None:
    # Color the pointcloud
    recolored_spheres = []
    for sphere in rgbd_spheres_list:
        recolored_sphere = deepcopy(sphere)
        recolored_sphere.paint_uniform_color(RGBD_COLOR)
        recolored_sphere.translate(RGBD_DELTA_POSITION)
        recolored_spheres.append(recolored_sphere)

    # Color the voxel grid
    recolored_voxel_mesh = deepcopy(voxel_mesh)
    recolored_voxel_mesh.paint_uniform_color(RECONSTRUCTION_COLOR)

    draw_geometries(
        [recolored_voxel_mesh] + recolored_spheres,
        output_path=output_dir / "rgbd_plus_voxel.png",
        viewpoint_path=viewpoint_path,
    )


if __name__ == "__main__":
    args = parse_arguments()

    depth_image_path = args.input_dir / f"{args.idx:04d}.wrist_depth.png"
    color_image_path = args.input_dir / f"{args.idx:04d}.wrist_rgb.png"
    intrinsics_path = args.input_dir / f"{args.idx:04d}.wrist_intrinsics.npy"
    pose_path = args.input_dir / f"{args.idx:04d}.wrist_pose.npy"
    map_path = args.input_dir / f"{args.idx:04d}.nvblox_map_static.nvblx"

    pca_params_path = args.output_dir / f"pca_params.pt"

    VIEWPOINT_FILE_PATH = SCRIPT_DIR / "input" / f"viewpoint.json"

    # % Voxel Grid  + PCA
    voxel_mesh, pca_specification = load_voxel_mesh_and_pca(
        map_path,
        recompute_pca=args.recompute_pca,
        pca_params_path=pca_params_path,
    )
    show_and_save_voxel_mesh(
        voxel_mesh, output_path=args.output_dir / "voxels.png", viewpoint_path=VIEWPOINT_FILE_PATH
    )

    # Warped Images
    depth_image, features_pca = load_images_warp_show_and_save(
        depth_image_path,
        color_image_path,
        output_dir=args.output_dir,
        pca_specification=pca_specification,
    )

    # Depth Pointcloud
    rgbd_spheres_list = show_and_save_depth_pointcloud(
        depth_image,
        features_pca,
        intrinsics_path,
        pose_path,
        output_dir=args.output_dir,
        viewpoint_path=VIEWPOINT_FILE_PATH,
    )

    # RGBD + Voxel
    save_rgbd_plus_voxel_fig(
        rgbd_spheres_list,
        voxel_mesh,
        output_dir=args.output_dir,
        viewpoint_path=VIEWPOINT_FILE_PATH,
    )

    print("Done")

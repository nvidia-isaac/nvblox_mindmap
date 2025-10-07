# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import dataclasses
import pathlib
from typing import Tuple

import cv2
import numpy as np
from nvblox_torch.indexing import get_voxel_center_grids
from nvblox_torch.mapper import Mapper
from nvblox_torch.visualization import get_voxel_mesh
import open3d as o3d
from pxr import Gf, Sdf, Usd, UsdGeom, Vt
import torch
import tqdm


@dataclasses.dataclass
class PCASpecification:
    projection_matrix: torch.Tensor
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor


def open3d_to_usd(mesh: o3d.geometry.TriangleMesh) -> Usd.Stage:
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)

    # Create a new USD stage
    stage = Usd.Stage.CreateInMemory()

    # Set the default prim
    default_prim: Usd.Prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World")).GetPrim()
    stage.SetDefaultPrim(default_prim)

    # Create a Mesh prim at the root
    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/reconstruction"))

    # Set the points (vertices)
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])

    # Set the face vertex indices
    faceVertexIndices = [int(i) for tri in faces for i in tri]
    mesh.CreateFaceVertexIndicesAttr(faceVertexIndices)

    # Set the face vertex counts (3 for each triangle)
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))

    # Add per-vertex RGB colors
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*rgb) for rgb in colors]))
    mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")

    # Add per-vertex normals
    mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in normals])
    mesh.SetNormalsInterpolation("vertex")

    return stage


def save_root_layer_to_usd(stage: Usd.Stage, usd_path: pathlib.Path) -> None:
    stage.GetRootLayer().Export(str(usd_path))


def visualize_color_mesh(mapper: Mapper):
    # Update color mesh
    mapper.update_color_mesh()
    mesh = mapper.get_color_mesh()
    mesh_o3d = mesh.to_open3d()
    mesh_o3d.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_o3d])


def visualize_feature_mesh(mapper: Mapper, pca_specification: PCASpecification | None = None):
    # Update feature mesh
    mapper.update_feature_mesh()
    feature_mesh = mapper.get_feature_mesh()
    feature_appearances = feature_mesh.vertex_appearances()
    feature_vertices_rgb, pca_specification = colors_from_features(
        feature_appearances, pca_specification
    )
    feature_mesh_rgb = o3d.geometry.TriangleMesh()
    feature_mesh_rgb.vertices = o3d.utility.Vector3dVector(feature_mesh.vertices().cpu().numpy())
    feature_mesh_rgb.vertex_colors = o3d.utility.Vector3dVector(feature_vertices_rgb.cpu().numpy())
    feature_mesh_rgb.triangles = o3d.utility.Vector3iVector(feature_mesh.triangles().cpu().numpy())
    feature_mesh_rgb.compute_vertex_normals()
    o3d.visualization.draw_geometries([feature_mesh_rgb])


def get_open3d_feature_cubes_mesh(
    mapper: Mapper, pca_specification: PCASpecification | None = None
) -> Tuple[o3d.geometry.TriangleMesh, PCASpecification]:
    blocks, indices = mapper.feature_layer_view().get_all_blocks()
    voxel_centers_list = get_voxel_center_grids(
        indices, mapper.feature_layer_view().voxel_size(), device="cuda"
    )
    features = []
    centers = []
    for feature_block, index, voxel_centers in tqdm.tqdm(zip(blocks, indices, voxel_centers_list)):
        tsdf_block = mapper.tsdf_layer_view().get_block_at_index(index)
        # Surface voxels
        tsdf = tsdf_block[..., 0]
        weight = tsdf_block[..., 1]
        valid_tsdf = torch.logical_and(tsdf < 0.00, weight > 0.01)
        # Extract surface features
        feature_weights = feature_block[..., -1]
        valid_feature_weights = feature_weights > 0.01
        # Combined mask
        valid_mask = torch.logical_and(valid_tsdf, valid_feature_weights)
        # Extract surface features
        surface_features = feature_block[valid_mask]
        centers.append(voxel_centers[valid_mask])
        features.append(surface_features)

    # Combine
    all_features = torch.cat(features, dim=0)
    all_centers = torch.cat(centers, dim=0)

    # Remove the weights
    all_features = all_features[..., :-1]

    # PCA
    colors, pca_specification = colors_from_features(all_features, pca_specification)

    # Visualize
    voxel_mesh = get_voxel_mesh(
        all_centers, mapper.feature_layer_view().voxel_size(), colors=colors
    )
    return voxel_mesh, pca_specification


def visualize_feature_cubes(mapper: Mapper):
    voxel_mesh, _ = get_open3d_feature_cubes_mesh(mapper)
    o3d.visualization.draw_geometries([voxel_mesh])


def get_pca_specification(features: torch.Tensor) -> PCASpecification:
    assert features.ndim == 2
    # Remove empty features when computing the basis
    valid_mask = ~torch.all(features == 0, dim=-1)
    tensor_nonzero = features[valid_mask].float()
    mean = tensor_nonzero.mean(0)
    with torch.no_grad():
        _, _, pca_v = torch.pca_lowrank(tensor_nonzero - mean, niter=5)
    # 3D Projection
    projection_matrix = pca_v[:, :3]
    # 1D Projection
    # projection_matrix = pca_v[:, :1]
    feature_vertices_rgb = features.float() @ projection_matrix
    lower_bound = torch.quantile(feature_vertices_rgb, 0.01, dim=0)
    upper_bound = torch.quantile(feature_vertices_rgb, 0.99, dim=0)
    return PCASpecification(
        projection_matrix=projection_matrix,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def colors_from_features(
    features: torch.Tensor, pca_specification: PCASpecification | None = None
) -> Tuple[torch.Tensor, PCASpecification]:
    assert features.ndim == 2
    # Compute PCA if not passed in.
    if pca_specification is None:
        pca_specification = get_pca_specification(features)
    # 3D Projection
    feature_vertices_rgb = features.float() @ pca_specification.projection_matrix
    feature_vertices_rgb = (feature_vertices_rgb - pca_specification.lower_bound) / (
        pca_specification.upper_bound - pca_specification.lower_bound
    )
    feature_vertices_rgb = torch.clamp(feature_vertices_rgb, 0, 1)
    # 1D Projection
    # feature_vertices_float = features.float() @ pca_specification.projection_matrix
    # feature_vertices_float = (feature_vertices_float - pca_specification.lower_bound) / (pca_specification.upper_bound - pca_specification.lower_bound)
    # feature_vertices_float = torch.clamp(feature_vertices_float, 0, 1)
    # from matplotlib import colormaps
    # cmap = colormaps["gnuplot"]
    # feature_vertices_rgb = cmap(feature_vertices_float.squeeze().cpu().numpy())[:, :3]
    # feature_vertices_rgb = torch.from_numpy(feature_vertices_rgb).to(features.device)
    assert feature_vertices_rgb.shape[0] == features.shape[0]
    assert feature_vertices_rgb.shape[1] == 3
    return feature_vertices_rgb, pca_specification


def set_viewpoint(visualizer: o3d.visualization.Visualizer, viewpoint_path: pathlib.Path) -> None:
    """Sets this example's inital viewpoint from file."""
    ctr = visualizer.get_view_control()
    if viewpoint_path.exists():
        param = o3d.io.read_pinhole_camera_parameters(viewpoint_path)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    else:
        print(f"Viewpoint file {viewpoint_path} does not exist. Not setting viewpoint.")


def draw_geometries(
    geometries: list[o3d.geometry.Geometry],
    viewpoint_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
) -> None:
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    if viewpoint_path is not None:
        set_viewpoint(visualizer, viewpoint_path)
    # Get screenshot
    if output_path is not None:
        screenshot = visualizer.capture_screen_float_buffer(do_render=True)
        screenshot = (np.asarray(screenshot) * 255.0).astype(np.uint8)
        screenshot = screenshot[:, :, :3]
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, screenshot)
    visualizer.run()
    visualizer.destroy_window()

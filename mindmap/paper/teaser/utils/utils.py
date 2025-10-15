# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import pathlib

import numpy as np
from nvblox_torch.indexing import get_voxel_center_grids
from nvblox_torch.mapper import Mapper
from nvblox_torch.visualization import get_voxel_mesh
import open3d as o3d
from pxr import Gf, Sdf, Usd, UsdGeom, Vt
import torch
import tqdm


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


def visualize_feature_mesh(mapper: Mapper):
    # Update feature mesh
    mapper.update_feature_mesh()
    feature_mesh = mapper.get_feature_mesh()
    feature_appearances = feature_mesh.vertex_appearances()
    feature_vertices_rgb = colors_from_features(feature_appearances)
    feature_mesh_rgb = o3d.geometry.TriangleMesh()
    feature_mesh_rgb.vertices = o3d.utility.Vector3dVector(feature_mesh.vertices().cpu().numpy())
    feature_mesh_rgb.vertex_colors = o3d.utility.Vector3dVector(feature_vertices_rgb.cpu().numpy())
    feature_mesh_rgb.triangles = o3d.utility.Vector3iVector(feature_mesh.triangles().cpu().numpy())
    feature_mesh_rgb.compute_vertex_normals()
    o3d.visualization.draw_geometries([feature_mesh_rgb])


def get_open3d_feature_cubes_mesh(mapper: Mapper) -> o3d.geometry.TriangleMesh:
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

    # PCA
    colors = colors_from_features(all_features)

    # Visualize
    voxel_mesh = get_voxel_mesh(
        all_centers, mapper.feature_layer_view().voxel_size(), colors=colors
    )
    return voxel_mesh


def visualize_feature_cubes(mapper: Mapper):
    voxel_mesh = get_open3d_feature_cubes_mesh(mapper)
    o3d.visualization.draw_geometries([voxel_mesh])


def colors_from_features(features: torch.Tensor) -> torch.Tensor:
    assert features.ndim == 2

    # Remove empty features when computing the basis
    valid_mask = ~torch.all(features == 0, axis=-1)
    tensor_nonzero = features[valid_mask].float()
    mean = tensor_nonzero.mean(0)
    with torch.no_grad():
        _, _, pca_v = torch.pca_lowrank(tensor_nonzero - mean, niter=5)
    projection_matrix = pca_v[:, :3]
    feature_vertices_rgb = features.float() @ projection_matrix
    lower_bound = torch.quantile(feature_vertices_rgb, 0.01, dim=0)
    upper_bound = torch.quantile(feature_vertices_rgb, 0.99, dim=0)
    feature_vertices_rgb = (feature_vertices_rgb - lower_bound) / (upper_bound - lower_bound)
    feature_vertices_rgb = torch.clamp(feature_vertices_rgb, 0, 1)
    assert feature_vertices_rgb.shape[0] == features.shape[0]
    assert feature_vertices_rgb.shape[1] == 3
    return feature_vertices_rgb

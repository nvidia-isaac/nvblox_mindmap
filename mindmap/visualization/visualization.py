# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import Optional, Tuple

import cv2
import matplotlib
import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch

from mindmap.data_loading.batching import unpack_batch
from mindmap.geometry.transforms import (
    look_at_to_transformation_matrix,
    transformation_trajectory_from_parts,
)
from mindmap.image_processing.pca import apply_pca_return_projection


class VideoWriter:
    def __init__(self, out_video_path: str, video_size: Tuple, fps: int = 15):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_size = video_size
        self.writer = cv2.VideoWriter(out_video_path, fourcc, fps, video_size)
        print(f"Writing video to: {out_video_path}")

    def add_image(self, img):
        # Permute to (BGR) and resize
        img_bgr = img.squeeze()[:, :, [2, 1, 0]]
        resized_img = cv2.resize(img_bgr.cpu().numpy(), self.video_size)
        self.writer.write(resized_img)

    def close(self):
        self.writer.release()


def get_sphere_mesh(
    position: torch.Tensor, radius: float, color: Optional[npt.NDArray] = None
) -> o3d.geometry.TriangleMesh:
    """Get a spere mesh at a position

    Args:
        position (torch.Tensor): 3D position tensor
        radius (float): radius of the sphere

    Returns:
        o3d.geometry.TriangleMesh: A mesh of the sphere.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.translate(position.cpu().numpy())
    if color:
        sphere.paint_uniform_color(color)
    return sphere


def get_segment_mesh(
    position_start: torch.Tensor, position_end: torch.Tensor, radius: float
) -> o3d.geometry.TriangleMesh:
    """Gets a mesh of a segment joining two positions

    Args:
        position_start (torch.Tensor): First end-point
        position_end (torch.Tensor): Second end-point
        radius (float): Radius of the cylinder mesh.

    Returns:
        o3d.geometry.TriangleMesh: A mesh of the connection.
    """
    center = (position_end - position_start) / 2.0 + position_start
    length = torch.norm(position_end - position_start)
    T_W_C = look_at_to_transformation_matrix(
        center_W=center,
        look_at_point_W=position_start,
        camera_up_W=torch.tensor([0.0, 0.0, 1.0], device="cuda"),
    )
    segment = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=length,
    )
    segment.compute_vertex_normals()
    segment.transform(T_W_C.cpu().numpy())
    return segment


def get_axis_mesh(
    T_W_C: Optional[npt.NDArray] = None, size: float = 1.0
) -> o3d.geometry.TriangleMesh:
    """Gets a coordinate frame as a mesh for visualization

    Args:
        T_W_C (npt.NDArray): Transform from the Camera coordinate frame
        (C) to the world coordinate frame (W).

    Returns:
        o3d.geometry.TriangleMesh: Mesh representing the axis.
    """
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if T_W_C is not None:
        axis.transform(T_W_C)
    return axis


def get_text_mesh(
    text: str, position: np.array, scale: float = 1e-3, color: npt.NDArray = (1, 1, 1)
):
    """Create text label mesh

    Args:
      text: Text
      position: Position of text
      scale: Scale of text
    """

    # Create at text mesh
    text_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=1.0).to_legacy()
    text_mesh.paint_uniform_color(color)

    # Scale the text
    text_mesh.transform(
        [
            [scale, 0, 0, position[0]],
            [0, scale, 0, position[1]],
            [0, 0, scale, position[2]],
            [0, 0, 0, 1],
        ]
    )
    return text_mesh


def get_trajectory_mesh(
    T_W_C_trajectory: torch.Tensor,
    sphere_radius: Optional[float] = None,
    segment_radius: Optional[float] = None,
    axis_size: Optional[float] = None,
    color: Optional[npt.NDArray] = None,
) -> o3d.geometry.TriangleMesh:
    """Gets a mesh representing the pose trajectory.

    Args:
        T_W_C_trajectory (torch.Tensor): The Nx4x4 pose trajectory.
        sphere_radius (float): The radius of spheres at each position
        segment_radius (float): The radius of the segments linking to positions.
        axis_size (float): The size of axis representing the poses.

    Returns:
        o3d.geometry.TriangleMesh: A mesh representing the trajectory.
    """
    assert T_W_C_trajectory.shape[1] == 4
    assert T_W_C_trajectory.shape[2] == 4
    colormap = matplotlib.colormaps.get_cmap("rainbow")
    trajectory_mesh = o3d.geometry.TriangleMesh()
    num_poses = T_W_C_trajectory.shape[0]
    use_generated_colors = True if color is None else False
    for start_idx in range(num_poses):
        # Extract parts
        pose_start = T_W_C_trajectory[start_idx, :, :]
        position_start = T_W_C_trajectory[start_idx, 0:3, 3]
        # Color for this segment
        if use_generated_colors:
            # If color is not set, use rainbow color
            color = np.array(colormap(start_idx / num_poses)[:3])
        # Segment mesh
        if segment_radius is not None and start_idx < num_poses - 1:
            position_end = T_W_C_trajectory[start_idx + 1, 0:3, 3]
            segment = get_segment_mesh(position_start, position_end, segment_radius)
            segment.paint_uniform_color(color)
            trajectory_mesh += segment
        # Sphere mesh
        if sphere_radius is not None:
            sphere = get_sphere_mesh(position_start, sphere_radius)
            sphere.paint_uniform_color(color)
            trajectory_mesh += sphere
        # Axis mesh
        if axis_size is not None:
            axis_mesh = get_axis_mesh(pose_start.cpu().numpy(), size=axis_size)
            trajectory_mesh += axis_mesh
    return trajectory_mesh


def visualize_axis(
    pose: npt.NDArray,
    visualizer: o3d.visualization.Visualizer,
    text=None,
    size=0.1,
    color=(1, 1, 1),
):
    """Add an axis mesh to the visualizer"""
    t_matrix = transformation_trajectory_from_parts(pose[:3].unsqueeze(0), pose[3:7].unsqueeze(0))[
        0
    ]
    axis_mesh = get_axis_mesh(t_matrix.cpu().numpy().astype(np.float64), size=size)
    visualizer.add_geometry(axis_mesh)
    if text:
        visualizer.add_geometry(get_text_mesh(text, pose[:3].cpu().numpy(), color=color))


def visualize_arrow(
    direction: npt.NDArray,
    origin: npt.NDArray,
    visualizer: o3d.visualization.Visualizer,
    text=None,
    size=0.1,
):
    """Add an arrow mesh to the visualizer"""
    # Create the arrow
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.003, cone_radius=0.006, cylinder_height=size * 0.8, cone_height=size * 0.2
    )

    from scipy.spatial.transform import Rotation as R

    def get_rotation_matrix_from_vectors(vec1, vec2):
        """Find the rotation matrix that aligns vec1 to vec2 using scipy"""
        vec1 = np.asarray(vec1).reshape(1, 3)
        vec2 = np.asarray(vec2).reshape(1, 3)
        rot, _ = R.align_vectors(vec2, vec1)
        return rot.as_matrix()

    # Find the rotation matrix to rotate the arrow to point along the direction
    # Default arrow points along +z, we want it to point along direction in xy
    default_dir = np.array([0, 0, 1])
    target_dir = np.array([direction[0], direction[1], 0.0])
    if np.linalg.norm(target_dir) > 1e-6:
        rot_mat = get_rotation_matrix_from_vectors(default_dir, target_dir)
        arrow.rotate(rot_mat, center=[0.0, 0.0, 0.0])
    arrow.translate(origin)
    arrow.paint_uniform_color([1.0, 1.0, 1.0])  # white

    visualizer.add_geometry(arrow, reset_bounding_box=False)
    if text:
        visualizer.add_geometry(get_text_mesh(text, origin))


def to_open3d_pointcloud(
    pointcloud: np.array,
    colors: np.array,
    max_distance: Optional[float] = None,
    compute_normals: bool = True,
):
    assert pointcloud.shape[1] == 3
    assert colors.shape[1] == 3
    # Visualizing the 3D point cloud using Open3D
    pcd_o3d = o3d.geometry.PointCloud()

    if max_distance is not None:
        mask = np.linalg.norm(pointcloud, axis=1) < max_distance
    else:
        mask = np.ones(pointcloud.shape[0], dtype=bool)
    pointcloud_filtered = pointcloud[mask]
    pcd_o3d.points = o3d.utility.Vector3dVector(pointcloud_filtered)
    # Add color to pointcloud
    if colors.shape[0] == 1:
        colors = np.array([colors[0] for _ in range(len(pointcloud_filtered))])
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors[mask] / 255.0)
    if compute_normals:
        pcd_o3d.estimate_normals()
    return pcd_o3d


def visualize_pointcloud(
    pointcloud: np.array,
    visualizer: o3d.visualization.Visualizer,
    colors: np.array,
    max_distance: Optional[float] = None,
    compute_normals: bool = True,
):
    """Add a pointcloud to the visualizer.

    Args:
      pointcloud: Points to visualize
      visualizer: O3d visualizer
      color: Point color
      max_distance: Remove points with distance-from-origin larger than this value
    """
    pcd_o3d = to_open3d_pointcloud(pointcloud, colors, max_distance, compute_normals)
    visualizer.add_geometry(pcd_o3d)
    return pcd_o3d


def visualize_voxel_grid(
    pointcloud: torch.Tensor,
    visualizer: o3d.visualization.Visualizer,
    colors: np.ndarray,
    voxel_size: float,
    max_distance: Optional[float] = None,
) -> None:
    pcd_o3d = to_open3d_pointcloud(pointcloud, colors, max_distance)
    voxel_grid_o3d = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_o3d, voxel_size=voxel_size)
    visualizer.add_geometry(voxel_grid_o3d)


def visualize_trajectory(
    trajectory: torch.Tensor,
    visualizer: o3d.visualization.Visualizer,
    color: list,
    segment_radius: float = 0.0035,
    axis_size: float = None,
):
    history_poses = transformation_trajectory_from_parts(
        trajectory[0, :3].reshape(1, 3), trajectory[0, 3:7].reshape(1, 4)
    )

    try:
        history_mesh = get_trajectory_mesh(
            history_poses, segment_radius=segment_radius, axis_size=axis_size, color=color
        )
    except:  # FIXME: handle more gracefully case when all points are the same
        return

    visualizer.add_geometry(history_mesh)


def compute_pca_basis_from_dataset(
    embodiment,
    data_loader,
    image_size,
    add_external_cam,
    data_type,
    feature_type,
    rgbd_min_depth_threshold,
    max_num_samples_for_pca=200,
):
    features = []
    for idx, batch in enumerate(data_loader):
        if idx >= max_num_samples_for_pca:
            break
        sample = unpack_batch(
            embodiment,
            batch,
            batch_size=1,
            image_size=image_size,
            num_history=1,
            data_type=data_type,
            feature_type=feature_type,
            add_external_cam=add_external_cam,
            rgbd_min_depth_threshold=rgbd_min_depth_threshold,
            device="cpu",
        )
        features.append(sample["vertex_features"].squeeze())
    _, pca_params = apply_pca_return_projection(torch.cat(features))
    return pca_params

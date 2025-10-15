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

from nvblox_torch.mapper import Mapper

from mindmap.paper.utils.utils import (
    get_open3d_feature_cubes_mesh,
    open3d_to_usd,
    save_root_layer_to_usd,
    visualize_color_mesh,
    visualize_feature_cubes,
    visualize_feature_mesh,
)


def convert_maps_to_usd(input_dir: pathlib.Path, visualize: bool = False, voxel_size: float = 0.01):
    """Convert nvblox maps to USD files.

    Args:
        input_dir: Directory containing nvblox maps
        visualize: Whether to visualize the first map
        voxel_size: Voxel size in meters
    """
    first_loop = True
    for map_path in sorted(input_dir.glob("*nvblox_map_static.nvblx")):
        print(map_path)

        # Load map
        mapper = Mapper(voxel_sizes_m=voxel_size)
        mapper.load_from_file(str(map_path), 0)
        print(
            f"Loaded map with {mapper.tsdf_layer_view().num_allocated_blocks()} tsdf blocks and {mapper.feature_layer_view().num_allocated_blocks()} feature blocks"
        )

        # Visualize meshes (on the first loop for debug)
        if visualize and first_loop:
            visualize_color_mesh(mapper)
            visualize_feature_mesh(mapper)
            visualize_feature_cubes(mapper)
            first_loop = False

        # Get the voxel mesh
        voxel_mesh, _ = get_open3d_feature_cubes_mesh(mapper)

        # Convert to USD
        print("Converting to USD")
        usd_path = map_path.with_suffix(".usda")
        stage = open3d_to_usd(voxel_mesh)

        print(f"Saving mesh to {usd_path}")
        save_root_layer_to_usd(stage, str(usd_path))
        print(f"Saved.")


def main():
    parser = argparse.ArgumentParser(description="Convert nvblox maps to USD files")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing nvblox maps"
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize the first map")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size in meters")

    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")

    convert_maps_to_usd(input_dir, args.visualize, args.voxel_size)


if __name__ == "__main__":
    main()

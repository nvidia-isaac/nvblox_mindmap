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
import glob
import pathlib

import numpy as np
from nvblox_python_tools.visualization.images import (
    clip_to_max,
    get_colorized_image,
    write_image_sequence_to_video,
)
import tqdm


def main(depth_dir: pathlib.Path, output_path: pathlib.Path):
    depth_image_paths = sorted(glob.glob(str(depth_dir / "frame*.npy")))

    depth_images = []
    for depth_path in tqdm.tqdm(depth_image_paths):
        depth = np.load(depth_path)
        depth = np.squeeze(depth)
        depth_clipped = clip_to_max(depth, max_value=3.0)
        depth_colorized = get_colorized_image(depth_clipped)
        depth_images.append(depth_colorized)

    write_image_sequence_to_video(depth_images, output_path, frame_rate=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs perceptor on a dataset and records some output."
    )
    parser.add_argument("depth_dir", type=pathlib.Path, help="Path to the input depth directory.")
    parser.add_argument("output_path", type=pathlib.Path, help="Path to the output path.")
    args = parser.parse_args()

    main(args.depth_dir, args.output_path)

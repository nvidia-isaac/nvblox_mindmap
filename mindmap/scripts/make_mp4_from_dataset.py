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
import os
import pathlib
import subprocess
from typing import Optional


def make_mp4_from_images(demo_dir, output_dir, image_name):
    output_path = output_dir / f"{image_name}.mp4"
    if os.path.exists(output_path):
        os.remove(output_path)
    subprocess.call(
        [
            "ffmpeg",
            "-framerate",
            "30",
            "-i",
            f"{demo_dir}/%04d.{image_name}.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            f"{output_path}",
        ]
    )


def get_args():
    parser = argparse.ArgumentParser(description="Create MP4 videos from image sequences")
    parser.add_argument(
        "--demo_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing the image sequences",
    )
    parser.add_argument(
        "--output_dir", type=pathlib.Path, help="Directory to save the output videos"
    )
    return parser.parse_args()


def main(demo_dir: pathlib.Path, output_dir: Optional[pathlib.Path] = None):
    if output_dir is None:
        output_dir = demo_dir

    # Check that ffmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed. Please install ffmpeg to create videos.")

    # Create output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    make_mp4_from_images(demo_dir, output_dir, "table_rgb")
    make_mp4_from_images(demo_dir, output_dir, "wrist_rgb")

    # Stack the two videos
    subprocess.call(
        [
            "ffmpeg",
            "-i",
            f"{output_dir}/table_rgb.mp4",
            "-i",
            f"{output_dir}/wrist_rgb.mp4",
            "-filter_complex",
            "hstack",
            f"{output_dir}/stacked.mp4",
        ]
    )


if __name__ == "__main__":
    args = get_args()
    main(args.demo_dir, args.output_dir)

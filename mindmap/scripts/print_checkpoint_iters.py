#!/usr/bin/env python3
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

import torch


def print_checkpoint_iters(checkpoint_names, checkpoint_dir):
    """Print iteration values from a list of checkpoint files."""
    for checkpoint_name in checkpoint_names:
        best_checkpoint_path = checkpoint_dir + "/" + checkpoint_name + "/best.pth"
        last_checkpoint_path = checkpoint_dir + "/" + checkpoint_name + "/last.pth"
        if not os.path.exists(best_checkpoint_path) or not os.path.exists(last_checkpoint_path):
            print(
                f"Warning: Checkpoint not found: {best_checkpoint_path} or {last_checkpoint_path}"
            )
            continue

        try:
            best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
            last_checkpoint = torch.load(last_checkpoint_path, map_location="cpu")
            best_iter_value = best_checkpoint.get("iter", "Not found")
            last_iter_value = last_checkpoint.get("iter", "Not found")
            print(
                f"{checkpoint_name}: {int(best_iter_value / 1000)}k (best), {int(last_iter_value / 1000)}k (last)"
            )
        except Exception as e:
            print(f"Error loading {best_checkpoint_path} or {last_checkpoint_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Print iteration values from checkpoint files")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="", help="Path to checkpoint directory"
    )
    parser.add_argument("checkpoints", nargs="+", help="Paths to checkpoint files")
    args = parser.parse_args()

    print_checkpoint_iters(args.checkpoints, args.checkpoint_dir)


if __name__ == "__main__":
    main()

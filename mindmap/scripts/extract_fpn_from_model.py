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

import torch

# NOTE(alexmillane, 2025-03-07): This script is used to extract the feature pyramid weights from a model checkpoint.
# This is useful while we work out how to properly get multi-scale features into the nvblox model.
# TODO(alexmillane, 2025-03-07): Remove this script once we have a proper multi-scale features in nvblox.


def extract_fpn_weights(model_path: str, output_path: str):
    """Extract feature pyramid weights from a model checkpoint and save them separately.

    Args:
        model_path: Path to the full model checkpoint
        output_path: Where to save the extracted FPN weights
    """
    # Load the model
    print(f"Loading model from {model_path}")
    model_dict = torch.load(model_path, map_location="cpu")

    # Extract the feature pyramid weights
    feature_pyramid_dict = {}
    for key in model_dict["weight"].keys():
        if "pyramid_network" in key:
            print(f"Extracting weights from {key}")
            new_key = key.split("pyramid_network.")[1]
            print(f"Saving as new key {new_key}")
            feature_pyramid_dict[new_key] = model_dict["weight"][key]

    # Save the weights
    print(f"Saving FPN weights to {output_path}")
    torch.save(feature_pyramid_dict, output_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract FPN weights from a model checkpoint")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the full model checkpoint"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Where to save the extracted FPN weights"
    )
    args = parser.parse_args()

    extract_fpn_weights(args.model_path, args.output_path)

#!/bin/env python3
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
import sys

import cv2
import numpy as np
import torch

from mindmap.image_processing.feature_extraction import FeatureExtractorType, get_feature_extractor
from mindmap.image_processing.pca import apply_pca, apply_pca_return_projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract image features")
    parser.add_argument(
        "--image_paths", nargs="+", type=str, required=True, help="Paths to the image files"
    )
    parser.add_argument(
        "--feature_type",
        type=FeatureExtractorType,
        required=True,
        help="Type of feature to extract",
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize the features")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the features"
    )
    parser.add_argument("--load_pca_path", type=str, help="Load the PCA parameters")
    parser.add_argument("--fpn_checkpoint", type=str, help="Path to the FPN checkpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_extractor = get_feature_extractor(args.feature_type, fpn_path=args.fpn_checkpoint)
    features = []

    pca_params = None
    if args.load_pca_path:
        pca_params = torch.load(args.load_pca_path)
        print(f"Loaded PCA parameters from {args.load_pca_path}")

    for image_path in args.image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        features = feature_extractor.compute(torch.from_numpy(image).unsqueeze(0).cuda())
        dim = features.shape[-1]

        if pca_params is None:
            _, pca_params = apply_pca_return_projection(features.view(-1, dim))
            torch.save(pca_params, os.path.join(args.output_dir, "pca_params.pt"))
            print(f"Saved PCA parameters to {os.path.join(args.output_dir, 'pca_params.pt')}")

        features_pca = apply_pca(features, pca_params)

        features_pca = (features_pca * 255).squeeze().cpu().numpy().astype(np.uint8)
        features_pca = features_pca.reshape(features_pca.shape[0], features_pca.shape[1], 3)

        # upscale pca image
        features_pca = cv2.resize(features_pca, (image.shape[1], image.shape[0]))

        # combine the two images
        combined_image = np.concatenate([image, features_pca], axis=1)
        if args.visualize:
            cv2.imshow("RGB", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imshow(image_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(-1)

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
            print(f"Saved feature image to {save_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

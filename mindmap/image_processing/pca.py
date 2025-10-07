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
from typing import Optional, Tuple

import torch
from torchtyping import TensorType


def apply_pca_return_projection(
    tensor_flat: TensorType[2, "num_samples", "d"],
    projection_matrix: Optional[TensorType] = None,
    lower_bound: Optional[TensorType] = None,
    upper_bound: Optional[TensorType] = None,
    num_iterations: int = 5,
    target_dimension: int = 3,
) -> Tuple[
    TensorType[2, "num_samples", "target_dimension_d"], Tuple[TensorType, TensorType, TensorType]
]:
    """
    Perform Principal Component Analysis (PCA) to reduce the dimensionality ("d") of a
    two dimensional tensor.
    Optionally accepts a pre-computed projection matrix and bounds for normalization.
    If not provided, these will be calculated from the input tensor.

    Args:
        tensor_flat (TensorType[..., "d"]): The input multichannel tensor to be downsampled.
        target_dimension (int): The number of dimensions to reduce the tensor to.
        projection_matrix (Optional[TensorType]): Precomputed PCA projection matrix. If None,
            a PCA projection matrix is calculated using the input tensor.
        lower_bound (Optional[TensorType]): Lower bounds for normalizing the reduced tensor.
            If None, the bounds are calculated from the input tensor.
        upper_bound (Optional[TensorType]): Upper bounds for normalizing the reduced tensor.
            If None, the bounds are calculated from the input tensor.
        num_iterations (int): The number of iterations for the PCA algorithm. Default is 5.

    Returns:
        Tuple[TensorType[..., "d"], Tuple[TensorType, TensorType, TensorType]]:
            A tuple where the first element is the tensor projected into the reduced-dimensional space,
            and the second element is a tuple containing the projection matrix, lower bounds,
            and upper bounds used for normalization.
    """
    # Modified from https://github.com/pfnet-research/distilled-feature-fields/blob/master/train.py

    if projection_matrix is None:
        # Remove empty features when computing the basis
        valid_mask = ~torch.all(tensor_flat == 0, axis=-1)
        tensor_nonzero = tensor_flat[valid_mask]

        mean = tensor_nonzero.mean(0)
        with torch.no_grad():
            _, _, V = torch.pca_lowrank(tensor_nonzero - mean, niter=num_iterations)
        projection_matrix = V[:, :target_dimension]
    low_rank = tensor_flat @ projection_matrix
    if lower_bound is None:
        lower_bound = torch.quantile(low_rank, 0.01, dim=0)
    if upper_bound is None:
        upper_bound = torch.quantile(low_rank, 0.99, dim=0)

    low_rank = (low_rank - lower_bound) / (upper_bound - lower_bound)
    low_rank = torch.clamp(low_rank, 0, 1)
    return low_rank, (projection_matrix, lower_bound, upper_bound)


def apply_pca(
    tensor_flat: TensorType[2, "num_samples", "d"],
    pca_parameters: Tuple[TensorType, TensorType, TensorType],
) -> TensorType[2, "num_samples", "target_dimension_d"]:
    """
    Apply PCA-based dimensionality reduction using precomputed PCA parameters.

    Args:
        tensor_flat (TensorType[..., "d"]): The input multichannel tensor to be downsampled.
        target_dimension (int): The number of dimensions to reduce the tensor to. Default is 3.
        pca_parameters: Precomputed PCA parameters, including the projection matrix,
            lower bounds, and upper bounds.
        num_iterations (int): The number of iterations for the PCA algorithm. Default is 5.

    Returns:
        TensorType[..., "target_dimension_d"]: The tensor projected into the reduced-dimensional space.
    """
    projection_matrix, lower_bound, upper_bound = pca_parameters
    return apply_pca_return_projection(tensor_flat, projection_matrix, lower_bound, upper_bound)[0]

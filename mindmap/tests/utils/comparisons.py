# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import torch

from mindmap.data_loading.batching import unpack_batch
from mindmap.data_loading.data_types import DataType
from mindmap.embodiments.embodiment_base import EmbodimentBase
from mindmap.image_processing.feature_extraction import FeatureExtractorType


def quaternion_is_close(quat1, quat2, rtol=1e-3, atol=1e-5):
    return (
        torch.isclose(quat1, quat2, rtol=rtol, atol=atol).all()
        or torch.isclose(quat1, -quat2, rtol=rtol, atol=atol).all()
    )


def tensors_are_close(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    name: str = "",
    verbose=False,
    rtol=1e-5,
    atol=1e-8,
    allowed_outlier_fraction=0,
):
    """Return True if tensors are close. If not, an error message + diff will be printed"""

    if tensor1.shape != tensor2.shape:
        if verbose:
            print(f"Tensor shape mismatch: {name}, {tensor1.shape} != {tensor2.shape}")
        return False

    if tensor1.dtype != tensor2.dtype:
        if verbose:
            print(f"Tensor type mismatch: {name}")
        return False

    inlier_mask = torch.isclose(tensor1, tensor2, rtol, atol)
    num_inliers = torch.sum(inlier_mask)
    outlier_fraction = 1.0 - num_inliers / tensor1.numel()
    if outlier_fraction > allowed_outlier_fraction:
        print(
            f"Tensor comparison failed: {name} with atol={atol}, rtol={rtol}. Outlier fraction={outlier_fraction}. Printing largest diffs:"
        )
        if tensor1.dtype == torch.bool:
            diff = tensor1[~inlier_mask] ^ tensor2[~inlier_mask]
        else:
            diff = torch.sort(tensor1[~inlier_mask] - tensor2[~inlier_mask], descending=True).values
        print(f"{diff}")
        return False
    else:
        return True


def datasets_are_close(
    embodiment: EmbodimentBase,
    dataloader1: str,
    dataloader2: str,
    batch_size: int,
    num_batches_to_compare: int = None,
    data_type: DataType = DataType.MESH,
    feature_type: FeatureExtractorType = FeatureExtractorType.RGB,
    verbose: bool = True,
):
    """Return True if the two datasets are close, i.e. if all tensors within are
    close. num_batches_to_compare can be used when the dataset sizes are different.

    Args:
        dataloader1: First dataloader to compare.
        dataloader2: Second dataloader to compare.
        batch_size: Batch size to use for comparison.
        num_batches_to_compare: Number of batches to compare. Use None to compare all batches.
        data_type: Data type to use for comparison.
        verbose: Whether to print verbose output.
    """
    if num_batches_to_compare is None:
        assert len(dataloader1) == len(dataloader2), "Need same number of frames to compare."
        num_batches_to_compare = len(dataloader1)

    assert num_batches_to_compare > 0, "Need more than zero frames to compare."

    data1_iter = iter(dataloader1)
    data2_iter = iter(dataloader2)

    # Tolerances used when comparing different items in the dataset
    # Condition for closeness: |tensor1 - tensor2| <= rtol * |tensor2| + atol
    item_to_tolerance = {
        # 1mm tolerance for vertex positions. 1% Outliers are accepted
        "vertices": {"atol": 1e-3, "rtol": 0.0, "allowed_outlier_fraction": 1e-2},
        "pcds": {"atol": 1e-3, "rtol": 0.0, "allowed_outlier_fraction": 1e-2},
        # It is difficult to make rendering entirely deterministic
        # across different architectures, so we allow for small
        # deviations in feature values
        "vertex_features": {"atol": 0.05, "rtol": 0.0, "allowed_outlier_fraction": 1e-2},
        "vertices_valid_mask": {"atol": 0.0, "rtol": 0.0, "allowed_outlier_fraction": 1e-6},
        "rgbs": {"atol": 0.05, "rtol": 0.0, "allowed_outlier_fraction": 1e-2},
        # 1mm tolerance for gripper predictions (but no outliers allowed)
        "gt_gripper_pred": {"atol": 1e-3, "rtol": 0.0, "allowed_outlier_fraction": 0.0},
    }
    # We allow for 5% of batches to be different
    MIN_EQUAL_BATCHES_RATIO = 0.95

    num_equal_batches = 0
    for i in range(0, num_batches_to_compare):
        if i >= num_batches_to_compare:
            break

        # Reset the seed to make random sampling reproducible. We need to reset it for both iterators
        # since they may belong to the same dataset, depending in which test we're running
        torch.manual_seed(0)
        data1 = next(data1_iter)
        torch.manual_seed(0)
        data2 = next(data2_iter)

        batch1 = unpack_batch(
            embodiment,
            data1,
            batch_size=batch_size,
            image_size=(512, 512),
            num_history=3,
            data_type=data_type,
            feature_type=feature_type,
            add_external_cam=True,
        )
        batch2 = unpack_batch(
            embodiment,
            data2,
            batch_size=batch_size,
            image_size=(512, 512),
            num_history=3,
            data_type=data_type,
            feature_type=feature_type,
            add_external_cam=True,
        )

        if not batch1.keys() == batch2.keys():
            print(f"Batch key mismatch: keys1: {batch1.keys()} keys2: {batch2.keys()}")
            return False

        num_valid = 0
        batch_is_equal = True
        for key in batch1:
            if batch1[key] is not None:
                rtol = 1e-5
                atol = 1e-5
                allowed_outlier_fraction = 0
                if key in item_to_tolerance:
                    atol = item_to_tolerance[key]["atol"]
                    rtol = item_to_tolerance[key]["rtol"]
                    allowed_outlier_fraction = item_to_tolerance[key]["allowed_outlier_fraction"]
                batch_is_equal &= tensors_are_close(
                    batch1[key],
                    batch2[key],
                    name=key,
                    verbose=verbose,
                    rtol=rtol,
                    atol=atol,
                    allowed_outlier_fraction=allowed_outlier_fraction,
                )
                num_valid += 1

        assert num_valid > 0, "Found empty batch"
        if not batch_is_equal:
            print(f"DIFF DETECTED IN DATASET BATCH COMPARISON FOR FRAME {i}")
        num_equal_batches += int(batch_is_equal)

    # Check the ration of batches that are equal
    ratio_of_equal_batches = num_equal_batches / float(num_batches_to_compare)
    datasets_are_close = ratio_of_equal_batches >= MIN_EQUAL_BATCHES_RATIO
    print(
        f"Datasets are close: {datasets_are_close}, Ratio of equal batches: {ratio_of_equal_batches:.2f}."
    )

    return datasets_are_close

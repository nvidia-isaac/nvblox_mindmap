# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import os
import pickle

from PIL import Image
import einops
from nvblox_torch.constants import constants
import pytest
import torch
from torchvision.transforms.functional import pil_to_tensor
import zstandard

from mindmap.image_processing.feature_extraction import FeatureExtractorType, get_feature_extractor
from mindmap.tests.utils.comparisons import tensors_are_close
from mindmap.tests.utils.constants import TestDataLocations


def _load_test_image():
    """Load a test data image"""
    image_pil = Image.open(TestDataLocations.test_image_filepath).convert("RGB")
    image = pil_to_tensor(image_pil).to(dtype=torch.float32, device="cuda").unsqueeze(0)

    return einops.rearrange(image, "b c w h -> b w h c") / 255.0


def _test_feature_extractor_nopadding(extractor_type: FeatureExtractorType):
    """Test that the feture extractor produces sensible results"""

    desired_size = (123, 123)
    extractor = get_feature_extractor(
        extractor_type, desired_output_size=desired_size, pad_to_nvblox_dim=False
    )
    image = _load_test_image()

    feature_image = extractor.compute(image)
    assert feature_image.shape[0] == 1
    assert feature_image.shape[1] == desired_size[0]
    assert feature_image.shape[2] == desired_size[1]
    assert feature_image.shape[3] == extractor.embedding_dim()
    assert not torch.all(feature_image == 0), "Expect non-zero entries"


def _test_feature_extractor_padding(extractor_type: FeatureExtractorType):
    """Test that the feture extractor produces sensible results"""

    desired_size = (123, 123)
    extractor = get_feature_extractor(
        extractor_type, desired_output_size=desired_size, pad_to_nvblox_dim=True
    )
    image = _load_test_image()
    feature_image = extractor.compute(image)
    assert feature_image.shape[0] == 1
    assert feature_image.shape[1] == desired_size[0]
    assert feature_image.shape[2] == desired_size[1]
    assert feature_image.shape[3] == constants.feature_array_num_elements()

    num_padded = constants.feature_array_num_elements() - extractor.embedding_dim()
    assert extractor.num_excess_features() == num_padded

    if num_padded > 0:
        assert torch.all(
            feature_image[:, :, :, -num_padded:] == 0
        ), "Expect zero entries at the end"


def _read_zst(file_path: str):
    """Read and unpickle and zstd file"""
    dctx = zstandard.ZstdDecompressor()
    with open(file_path, "rb") as infile:
        with dctx.stream_reader(infile) as reader:
            sample = pickle.load(reader)
    return sample


def _write_zstd(data: torch.Tensor, file_path: str):
    """Pickle and compress a file"""
    compressor = zstandard.ZstdCompressor(level=1)
    with open(file_path, "wb") as outfile:
        outfile.write(compressor.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)))


def _get_baseline_feature_path(extractor_type):
    """Get the test data dir path for the given feature type"""
    return os.path.join(TestDataLocations.test_data_dir, f"{extractor_type.name}.zst")


def _test_feature_extractor_regression(
    extractor_type: FeatureExtractorType, generate_baseline: bool
):
    """Check that computed feature are similar to ones stored in test data dir"""

    torch.manual_seed(0)
    # Reduce size to save disk space
    extractor = get_feature_extractor(
        extractor_type, desired_output_size=(16, 16), pad_to_nvblox_dim=False
    )
    image = _load_test_image()

    # We scale down the feature image to save space on disk
    feature_image = extractor.compute(image).squeeze()

    # Optionally store the features to disk
    if generate_baseline:
        _write_zstd(feature_image, _get_baseline_feature_path(extractor_type))

    baseline_feature_image = _read_zst(_get_baseline_feature_path(extractor_type))

    # Absolute tolerance set to 5% of the mean feature value (excluding zeros)
    atol = torch.mean(torch.abs(feature_image[feature_image != 0])) / 20
    assert tensors_are_close(
        feature_image, baseline_feature_image, name=extractor_type.name, atol=atol, verbose=True
    )


def test_feature_extractors_nopadding():
    """Test all feature extractors without padding"""
    for extractor_type in FeatureExtractorType:
        _test_feature_extractor_nopadding(extractor_type)


def test_feature_extractors_padding():
    """Test all feature extractors with padding"""
    for extractor_type in FeatureExtractorType:
        _test_feature_extractor_padding(extractor_type)


def test_feature_extractors_regression(generate_baseline_arg):
    """Regression test for all feature extractors"""
    for extractor_type in FeatureExtractorType:
        _test_feature_extractor_regression(extractor_type, generate_baseline_arg)

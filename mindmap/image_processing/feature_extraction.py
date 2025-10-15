# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from abc import ABC, abstractmethod
from enum import Enum
import math
from typing import Optional, Tuple

import clip
from clip.model import ModifiedResNet
import einops
from nvblox_torch.constants import constants
import torch
import torch.distributed as dist
from torchvision import transforms as T
from torchvision.ops import FeaturePyramidNetwork

from mindmap.common_utils.torch_utils import AllowMatMulTf32
from mindmap.model_utils.distributed_training import get_rank

# Prevent the following error:
# HTTPError: HTTP Error 403: rate limit exceeded
# taken from https://github.com/pytorch/pytorch/issues/61755#issuecomment-885801511
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


class FeatureExtractorType(Enum):
    """Enum for identifying feature extractors. See the respective FeatureExtractor class for more details."""

    CLIP_RESNET50_FPN = "clip_resnet50_fpn"
    RADIO_V25_B = "radio_v25_b"
    DINO_V2_VITS14 = "dino_v2_vits14"
    RGB = "rgb"


def assert_square_and_batched_image(image):
    """Assert that the image is a square and batched"""
    assert image.ndim == 4, "Expected BxHxWxC"
    assert image.shape[1] == image.shape[2], "Need square images"


def get_nvblox_feature_dim(feature_extractor_type: FeatureExtractorType):
    """Get the embedding dimension for the feature extractor of specified type."""

    if feature_extractor_type == FeatureExtractorType.CLIP_RESNET50_FPN:
        return ClipResNet50FpnFeatureExtractor.embedding_dim()
    elif feature_extractor_type == FeatureExtractorType.RADIO_V25_B:
        return RadioV25BFeatureExtractor.embedding_dim()
    elif feature_extractor_type == FeatureExtractorType.DINO_V2_VITS14:
        return DinoV2Vits14FeatureExtractor.embedding_dim()
    elif feature_extractor_type == FeatureExtractorType.RGB:
        return RgbFeatureExtractor.embedding_dim()
    else:
        raise ValueError(f"Invalid feature extractor type: {feature_extractor_type}")


def get_feature_extractor(
    feature_extractor_type: FeatureExtractorType,
    feature_image_size: Optional[Tuple[int, int]] = None,
    desired_output_size: Optional[Tuple[int, int]] = None,
    pad_to_nvblox_dim: bool = False,
    fpn_path: Optional[str] = None,
):
    """Get a feature extractor based on the type

    Args:
        feature_extractor_type, FeatureExtractorType : The type of feature extractor to get.
        feature_image_size, Optional[Tuple[int, int]] : The desired dimension of the feature extractor output. If provided,
            the input image will be resized such that the output feature image has the desired dimension.
        desired_output_size, Optional[Tuple[int, int]] : If a value is provided, the feature image output from the extractor
            will be upscaled/downscaled to match the desired size.
        pad_to_nvblox_dim : If True, the features will be padded with zeros to support nvblox's feature dimension.
        fpn_path, Optional[str] : For CLIP-FPN. If provided, the FPN will be loaded from the given path.
    """
    if feature_extractor_type == FeatureExtractorType.CLIP_RESNET50_FPN:
        return ClipResNet50FpnFeatureExtractor(
            feature_image_size=feature_image_size,
            desired_output_size=desired_output_size,
            pad_to_nvblox_dim=pad_to_nvblox_dim,
            fpn_path=fpn_path,
        )
    elif feature_extractor_type == FeatureExtractorType.RADIO_V25_B:
        return RadioV25BFeatureExtractor(
            feature_image_size=feature_image_size,
            desired_output_size=desired_output_size,
            pad_to_nvblox_dim=pad_to_nvblox_dim,
        )
    elif feature_extractor_type == FeatureExtractorType.DINO_V2_VITS14:
        return DinoV2Vits14FeatureExtractor(
            feature_image_size=feature_image_size,
            desired_output_size=desired_output_size,
            pad_to_nvblox_dim=pad_to_nvblox_dim,
        )
    elif feature_extractor_type == FeatureExtractorType.RGB:
        return RgbFeatureExtractor(
            feature_image_size=feature_image_size,
            desired_output_size=desired_output_size,
            pad_to_nvblox_dim=pad_to_nvblox_dim,
        )
    else:
        raise ValueError(f"Invalid feature extractor type: {feature_extractor_type}")


def scale_image(tensor: torch.Tensor, target_size: Tuple[int, int], mode: str = "bilinear"):
    """
    Upscales a given tensor to the specified target size using interpolation.
    This function expects an input tensor in channel-first format `[B, C, H, W]`

    Args:
        tensor (torch.Tensor): The input tensor of shape `[B, C, H, W]` to scale.
        target_size (Tuple[int, int]): The desired output size.
        mode (str, optional): Interpolation mode. Default is `'bilinear'`.

    Returns:
        Tuple[torch.Tensor, int]:
            - The upscaled tensor of shape `[H_new, W_new, C]`.
            - The scale factors for height and width `(scale_h, scale_w)`.
    """
    assert tensor.ndim == 4
    tensor = torch.nn.functional.interpolate(
        tensor, size=target_size, mode=mode, align_corners=False
    )
    return tensor


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors"""

    def __init__(
        self,
        feature_image_size: Optional[Tuple[int, int]] = None,
        pad_to_nvblox_dim: bool = False,
        desired_output_size: Optional[Tuple[int, int]] = None,
    ):
        """Constructor:

        Args:
            feature_image_size, Optional[Tuple[int, int]] : The desired dimension of the feature extractor output. If provided,
                the input image will be resized such that the output feature image has the desired dimension.
            pad_to_nvblox_dim : If True, the features will be padded with zeros to support nvblox's feature dimension.
            desired_output_size, Optional[Tuple[int, int]] : If a value is provided, the feature image output from the extractor
                will be upscaled/downscaled to match the desired size.
        """
        self.feature_image_size = feature_image_size
        self.pad_to_nvblox_dim = pad_to_nvblox_dim
        self.desired_output_size = desired_output_size
        self.model = self.load()
        if self.model is not None:
            self.model.cuda().eval()
            for p in self.model.parameters():
                p.requires_grad = False

        assert self.embedding_dim() <= constants.feature_array_num_elements(), (
            f"Embedding dim: {self.embedding_dim()} is greater than nvblox's max feature size: "
            f"{constants.feature_array_num_elements()}. Rebuild nvblox with a larger feature size."
        )

    def train_dataset_mean_and_std(self):
        """Return the mean and std of the dataset used to train the extractor.
        Here we return dummy values that don't normalize.
        Override in subclass if normalization is desired."""
        return torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 1.0])

    def compute(self, rgb: torch.Tensor):
        """Main entry function for computing features.

        Args:
            rgb (b, h, w, 3): Input RGB image tensor.
        Returns:
            features (b, h, w, F) Computed features with additional zero-padding to support nvblox's feature dimension.
        """
        assert rgb.ndim == 4
        assert rgb.shape[3] == 3

        # Normalize and resize the image to prepare for the model
        rgb_channel_first = self.preprocess_image(rgb, self.train_dataset_mean_and_std())

        # Compute the features
        features_bchw = self._extract_features_impl(rgb_channel_first)

        # If we want to resize the output features to a specific size, do so
        if self.desired_output_size is not None:
            features_bchw = scale_image(features_bchw, self.desired_output_size)
        # Convert to channel last
        features_bhwc = einops.rearrange(features_bchw, "b c h w -> b h w c")

        # Nvblox feature mapper expects a dimension of a certain size, so we pad with zeros if needed
        if self.pad_to_nvblox_dim:
            features_bhwc = self._get_zero_padded_features(features_bhwc)
        return features_bhwc

    def _get_zero_padded_features(self, features_bhwc: torch.Tensor):
        """Return a padded version of features that has the correct dimensions for integration with nvblox's mapper"""
        assert_square_and_batched_image(features_bhwc)
        assert (
            features_bhwc.shape[3] == self.embedding_dim()
        ), f"Features have incorrect embedding dimension: {features_bhwc.shape[3]} != {self.embedding_dim()}"

        n_batches = features_bhwc.shape[0]
        feature_side_length = features_bhwc.shape[1]
        zeros = torch.zeros(
            n_batches, feature_side_length, feature_side_length, self.num_excess_features()
        ).to(features_bhwc.device)
        return torch.cat((features_bhwc, zeros), dim=3)

    def num_excess_features(self):
        """Return number of zeros that we need to append to the end of each feature in order to
        comply with nvblox lib"""
        num_excess = constants.feature_array_num_elements() - self.embedding_dim()
        assert num_excess >= 0, (
            f"Embedding dim: {self.embedding_dim()} is less than nvblox's max feature size: "
            f"{constants.feature_array_num_elements()}. Rebuild nvblox with a larger feature size."
        )
        return num_excess

    def preprocess_image(
        self, rgb_bhwc: torch.Tensor, mean_and_std: Tuple[torch.Tensor, torch.Tensor]
    ):
        """Preprocess an image for feature extraction."""
        mean, std = mean_and_std
        # Convert to float if needed
        if rgb_bhwc.dtype == torch.uint8:
            rgb_bhwc = rgb_bhwc.float() / 255.0
        else:
            assert (
                torch.max(rgb_bhwc) <= 1.0 and torch.min(rgb_bhwc) >= 0.0
            ), "Image should be normalized to [0, 1]"

        # Normalize with mean and std
        rgb_bhwc = (rgb_bhwc - mean.to(device="cuda")) / std.to(device="cuda")

        # Convert to channel first
        rgb_bchw = einops.rearrange(rgb_bhwc, "b h w c -> b c h w")

        # Scale
        if self.feature_image_size is not None:
            required_input_size = (
                self.feature_image_size[0] * self.model_downscale_factor(),
                self.feature_image_size[1] * self.model_downscale_factor(),
            )
        else:
            required_input_size = self.model_input_size()
        # Check that we can handle the desired image size
        assert required_input_size[0] % self.model_input_size()[0] == 0
        assert required_input_size[1] % self.model_input_size()[1] == 0
        rgb_bchw = scale_image(rgb_bchw, required_input_size)

        return rgb_bchw

    @abstractmethod
    def embedding_dim():
        """Number of active elements in a feature vector"""
        pass

    @abstractmethod
    def model_input_size(self) -> Tuple[int, int]:
        """Native input size of the extractor network. Integer multiples of this size can also be used."""
        pass

    @abstractmethod
    def model_output_size(self) -> Tuple[int, int]:
        """Output size of the extractor network"""
        pass

    def model_downscale_factor(self) -> int:
        """Factor by which the model downscales the input image to form the feature image"""
        input_size = self.model_input_size()
        output_size = self.model_output_size()
        assert input_size[0] % output_size[0] == 0
        assert input_size[1] % output_size[1] == 0
        assert input_size[0] / output_size[0] == input_size[1] / output_size[1]
        return int(input_size[0] / output_size[0])

    @abstractmethod
    def _extract_features_impl(self, rgb: torch.Tensor):
        """Implement this function to compute the features"""
        pass

    @abstractmethod
    def load_model():
        """Implement this function to load and return the model"""
        pass

    def load(self):
        """Load the model. This typically fetches the model from a remote and can thus be used to pre-cache the model."""

        # If we are running in distributed mode, we first let the main process load the model to avoid race conditions
        # when pulling from a remote.
        if dist.is_available() and dist.is_initialized():
            if get_rank() == 0:
                self.load_model()
            torch.distributed.barrier()

        return self.load_model()


class RadioFeatureExtractorBase(FeatureExtractor):
    """Base class for RADIO featue extraction.
    load_model() and embedding_dim() must be implemented in subclass."""

    def __init__(
        self,
        feature_image_size: Optional[Tuple[int, int]] = None,
        pad_to_nvblox_dim: bool = False,
        desired_output_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(
            feature_image_size=feature_image_size,
            pad_to_nvblox_dim=pad_to_nvblox_dim,
            desired_output_size=desired_output_size,
        )

    @torch.no_grad()
    def _extract_features_impl(self, rgb_bchw: torch.Tensor):
        # Compute features
        with AllowMatMulTf32():
            _, features = self.model(rgb_bchw)

        # Reshape output into image
        output_size = int(math.sqrt(features.shape[1]))
        num_batches = rgb_bchw.shape[0]
        return einops.rearrange(
            features.view(num_batches, output_size, output_size, -1), "b h w c -> b c h w"
        )

    def model_input_size(self):
        return (256, 256)

    def model_output_size(self):
        return (16, 16)


class RadioV25BFeatureExtractor(RadioFeatureExtractorBase):
    """RADIO feature extractor for the v2.5-b model.
    For more details, see https://huggingface.co/nvidia/RADIO"""

    def __init__(
        self,
        feature_image_size: Optional[Tuple[int, int]] = None,
        pad_to_nvblox_dim: bool = False,
        desired_output_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(
            feature_image_size=feature_image_size,
            pad_to_nvblox_dim=pad_to_nvblox_dim,
            desired_output_size=desired_output_size,
        )

    @staticmethod
    def embedding_dim():
        return 768

    @staticmethod
    def load_model():
        """Load and return radio model from torchhub"""
        model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version="radio_v2.5-b",
            progress=True,
            pretrained=True,
            skip_validation=True,
        )
        return model


class ClipResNet50FpnFeatureExtractor(FeatureExtractor, torch.nn.Module):
    """CLIP Feature extractor using Resnet50 + feature pyramid network.
    TODO(dtingdahl): This extractor requires a pre-trained FPN which makes it cumbersome to use.
                     should be retired once a baseline using a non-FPN extractor has been established.
    NOTE(remos): We inherit from torch.nn.Module to include the FPN in the model graph and make it trainable.
    """

    def __init__(
        self,
        feature_image_size: Optional[Tuple[int, int]] = None,
        pad_to_nvblox_dim: bool = False,
        desired_output_size: Optional[Tuple[int, int]] = None,
        fpn_path: Optional[str] = None,
    ):
        torch.nn.Module.__init__(self)
        FeatureExtractor.__init__(
            self,
            feature_image_size=feature_image_size,
            pad_to_nvblox_dim=pad_to_nvblox_dim,
            desired_output_size=desired_output_size,
        )

        # Load the backbone from the CLIP model
        self.backbone = self.load_backbone(self.model)

        # Load the FPN (from a pre-trained model or initialized to random weights)
        self.pyramid_network = self.load_fpn(fpn_path, self.embedding_dim())

    @staticmethod
    def load_backbone(model):
        # Extract the visual layers from the model
        state_dict = model.state_dict()
        layers = tuple(
            [
                len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
                for b in [1, 2, 3, 4]
            ]
        )
        output_dim = state_dict["text_projection"].shape[1]
        heads = state_dict["visual.layer1.0.conv1.weight"].shape[0] * 32 // 64

        # Load the extracted visual layers into a modified resnet and load the weights.
        backbone = ModifiedResNetFeatures(layers, output_dim, heads).cuda()
        backbone.load_state_dict(model.visual.state_dict())

        # Freeze the backbone weights
        for p in backbone.parameters():
            p.requires_grad = False

        return backbone

    @staticmethod
    def load_fpn(fpn_path: str, embedding_dim: int):
        pyramid_network = (
            FeaturePyramidNetwork([64, 256, 512, 1024, 2048], embedding_dim).cuda().eval()
        )

        if fpn_path is not None:
            # Load the pre-trained FPN
            fpn_model = torch.load(fpn_path)
            pyramid_network.load_state_dict(fpn_model)
            # When loading the FPN, we want to keep the weights frozen.
            fix_fpn_weights = True
        else:
            # When not loading a pre-trained FPN,
            # we want to train the FPN jointly with the model.
            fix_fpn_weights = False

        for p in pyramid_network.parameters():
            p.requires_grad = not fix_fpn_weights

        return pyramid_network

    def train_dataset_mean_and_std(self):
        # CLIP was trained on the WebImageText dataset, so normalization constants are different here.
        WIT_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        WIT_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        return WIT_MEAN, WIT_STD

    @staticmethod
    def embedding_dim():
        return 120

    @torch.no_grad()
    def _extract_features_impl(self, rgb_bchw: torch.Tensor):
        # Extract the layers from CLIP backbone
        layer_outputs = self.backbone(rgb_bchw)

        # Pass through the FPN
        encoded = self.pyramid_network(layer_outputs)

        return encoded["res3"]

    def model_input_size(self):
        return (256, 256)

    def model_output_size(self):
        return (16, 16)

    @staticmethod
    def load_model():
        """Load and return clip model from torchhub"""
        model, _ = clip.load("RN50")
        return model


class ModifiedResNetFeatures(ModifiedResNet):
    """Modified ResNet for CLIP that allows for retrieval of features from intermediate layers in CLIP/resnet.
    Adapted from from the diffuser actor repo: https://github.com/nickgkan/3d_diffuser_actor"""

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__(layers, output_dim, heads, input_resolution, width)

    def forward(self, x: torch.Tensor):
        x = x.type(self.conv1.weight.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x0 = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {
            "res1": x0,
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4,
        }


class DinoV2Vits14FeatureExtractor(FeatureExtractor):
    """Feature extractor that uses the last layer of DINO_V2_VITS14"""

    def __init__(
        self,
        feature_image_size: Optional[Tuple[int, int]] = None,
        pad_to_nvblox_dim: bool = False,
        desired_output_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(
            feature_image_size=feature_image_size,
            pad_to_nvblox_dim=pad_to_nvblox_dim,
            desired_output_size=desired_output_size,
        )

    @staticmethod
    def embedding_dim():
        return 384

    def model_input_size(self):
        return (224, 224)

    def model_output_size(self):
        return (16, 16)

    @staticmethod
    def load_model():
        """Load and return dino model from torchhub"""
        model = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vits14")
        return model

    @torch.no_grad()
    def _extract_features_impl(self, rgb_bchw: torch.Tensor):
        features = self.model.get_intermediate_layers(rgb_bchw, n=1)[0]  # Get last layer features

        # Reshape output into image
        output_size = int(math.sqrt(features.shape[1]))
        num_batches = rgb_bchw.shape[0]
        return einops.rearrange(
            features.view(num_batches, output_size, output_size, -1), "b h w c -> b c h w"
        )

    def train_dataset_mean_and_std(self):
        # DINO_V2_VITS14 was trained on the ImageNet dataset, return the appropriate mean and std.
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
        return IMAGENET_MEAN, IMAGENET_STD


class RgbFeatureExtractor(FeatureExtractor):
    """Feature extractor that just returns a scaled version of the input RGB image"""

    def __init__(
        self,
        feature_image_size: Optional[Tuple[int, int]] = None,
        pad_to_nvblox_dim: bool = False,
        desired_output_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(
            feature_image_size=feature_image_size,
            pad_to_nvblox_dim=pad_to_nvblox_dim,
            desired_output_size=desired_output_size,
        )

    @staticmethod
    def embedding_dim():
        return 3

    def model_input_size(self):
        # We're scaling the RGB image to a size suitable for training
        return (32, 32)

    def model_output_size(self):
        return (32, 32)

    @staticmethod
    def load_model():
        pass

    @torch.no_grad()
    def _extract_features_impl(self, rgb_bchw: torch.Tensor):
        return rgb_bchw

import dgl.geometry as dgl_geo
import einops
import torch
from torch import nn
from torch.nn import functional as F

from mindmap.data_loading.data_types import DataType, includes_mesh
from mindmap.diffuser_actor.layers import FFWRelativeCrossAttentionModule, ParallelAttention
from mindmap.diffuser_actor.position_encodings import RotaryPositionEncoding3D
from mindmap.image_processing.feature_extraction import (
    FeatureExtractorType,
    get_feature_extractor,
    get_nvblox_feature_dim,
)
from mindmap.image_processing.image_mask_operations import downscale_mask


class ImageFeatureEmbedder(nn.Module):
    def __init__(self, in_features, out_features):
        super(ImageFeatureEmbedder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features).to(device="cuda")

    def forward(self, features_BxNxC: torch.Tensor):
        assert features_BxNxC.ndim == 3
        assert features_BxNxC.shape[-1] == self.linear.in_features
        encoded_features = self.linear(features_BxNxC)
        return encoded_features


class Encoder(nn.Module):
    def __init__(
        self,
        image_size=(256, 256),
        feature_image_size=(32, 32),
        embedding_dim=60,
        nhist=3,
        ngrippers=1,
        num_attn_heads=8,
        num_vis_ins_attn_layers=2,
        fps_subsampling_factor=5,
        data_type=DataType.RGBD,
        encode_openness=False,
        feature_type=FeatureExtractorType.CLIP_RESNET50_FPN,
        encoder_dropout=0.0,
        fpn_path=None,
        use_shared_feature_encoder=False,
    ):
        """Construct an Encoder.

        Args:
            image_size: Input image size. Either (512, 512) or (256, 256) or (128, 128)
            feature_image_size: The desired dimension of the feature extractor output.
            embedding_dim: The dimensionality of embedded tokens
            nhist: Gripper history length
            num_attn_heads:  Num attention heads for gripper<->context and gripper<->language
            num_vis_ins_attn_layers: Num layers for gripper<->languge
            fps_subsampling_factor: Amount of furthest-point-subsampling
            data_type: Type of input data
            encode_openness: If true, different learnable queries are used for gripper open/closed
            encoder_dropout: Dropout probability for attention modules
            fpn_path: Path to pre-trained feature pyramid network (only used for CLIP features)
            use_shared_feature_encoder: If true, use the same encoder and weights as for the RGB extractor for the mesh features.
        """
        super().__init__()
        assert image_size in [(128, 128), (256, 256), (512, 512)]

        self.image_size = image_size
        self.fps_subsampling_factor = fps_subsampling_factor
        self.encode_openness = encode_openness
        self.use_shared_feature_encoder = use_shared_feature_encoder

        if data_type == DataType.RGBD or data_type == DataType.RGBD_AND_MESH:
            # Feature extractor
            if (
                feature_type == FeatureExtractorType.CLIP_RESNET50_FPN
                and data_type != DataType.RGBD
            ):
                # Not setting an FPN is only allowed for RGBD type.
                # In the RGBD case, when the FPN is not provided, it is trained/loaded along with the rest of the model.
                assert fpn_path is not None, "FPN checkpoint required for CLIP extractor"

            # Frozen backbone
            self.feature_extractor = get_feature_extractor(
                feature_extractor_type=feature_type,
                feature_image_size=feature_image_size,
                pad_to_nvblox_dim=False,
                fpn_path=fpn_path,
            )

            # Encoder to transform features into a compatible embedding dimension
            self.image_feature_encoder = ImageFeatureEmbedder(
                in_features=get_nvblox_feature_dim(feature_type), out_features=embedding_dim
            )

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Current gripper learnable features
        # 1) Used if encode_openness=False
        #    Constant learnable query
        if not encode_openness:
            self.gripper_history_embed = nn.Embedding(nhist * ngrippers, embedding_dim)
        else:
            self.gripper_history_embed = None
        # 2) Used if encode_openness=True
        #    Maps gripper open/closed to a query. Note that because the input is a
        #    binary variable, the output switches between two query vectors. Therefore
        #    this linear layer is equivalent to two learnable queries, one for open,
        #    one for closed.
        if encode_openness:
            self.curr_open_close_encoder = nn.Linear(
                in_features=nhist * ngrippers, out_features=nhist * ngrippers * embedding_dim
            )
        else:
            self.curr_open_close_encoder = None
        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=3, dropout=encoder_dropout, use_adaln=False
        )

        # Goal gripper learnable features
        self.goal_gripper_embed = nn.Embedding(1, embedding_dim)

        # Instruction encoder
        self.instruction_encoder = nn.Linear(512, embedding_dim)

        # Attention from vision to language
        layer = ParallelAttention(
            num_layers=num_vis_ins_attn_layers,
            d_model=embedding_dim,
            n_heads=num_attn_heads,
            dropout=encoder_dropout,
            self_attention1=False,
            self_attention2=False,
            cross_attention1=True,
            cross_attention2=False,
        )
        self.vl_attention = nn.ModuleList([layer for _ in range(1) for _ in range(1)])

        nvblox_feature_dim = get_nvblox_feature_dim(feature_type)

        if includes_mesh(data_type):
            # We need an encoder for the mesh features to bring them down to the embedding dimension.
            # If requested, we use the same encoder and weights as for the RGB feature extractor. Otherwise create a distinct one.
            if use_shared_feature_encoder:
                self.reconstruction_encoder = None
            else:
                self.reconstruction_encoder = ImageFeatureEmbedder(
                    in_features=nvblox_feature_dim, out_features=embedding_dim
                )

        elif data_type == DataType.RGBD:
            # No encoder needed for RGBD data type
            self.reconstruction_encoder = None

        else:
            raise ValueError(f"Data type '{data_type}' is not implemented.")

    def forward(self):
        return None

    def encode_gripper_history(self, gripper_history, context_feats, context, curr_closedness):
        """
        Compute current gripper position features and positional embeddings.

        Args:
            - gripper_history: (B, nhist, ngrippers, 3+)

        Returns:
            - gripper_history_feats: (B, nhist, F)
            - gripper_history_pos: (B, nhist, F, 2)
        """
        return self._encode_gripper(
            gripper_history, self.gripper_history_embed, context_feats, context, curr_closedness
        )

    def encode_goal_gripper(self, goal_gripper, context_feats, context):
        """
        Compute goal gripper position features and positional embeddings.

        Args:
            - goal_gripper: (B, 3+)

        Returns:
            - goal_gripper_feats: (B, 1, F)
            - goal_gripper_pos: (B, 1, F, 2)
        """
        goal_gripper_feats, goal_gripper_pos, _ = self._encode_gripper(
            goal_gripper[:, None], self.goal_gripper_embed, context_feats, context
        )
        return goal_gripper_feats, goal_gripper_pos

    def _encode_gripper(self, gripper, gripper_embed, context_feats, context, curr_closedness):
        """
        Compute gripper position features and positional embeddings.

        Args:
            - gripper: (B, npt, ngrippers, 3+)
            - context_feats: (B, npt, C)
            - context: (B, npt, 3)

        Returns:
            - gripper_feats: (B, npt, F)
            - gripper_pos: (B, npt, F, 2)
        """
        # Learnable embedding for gripper
        if self.encode_openness:
            nhist = curr_closedness.shape[1]
            ngrippers = curr_closedness.shape[2]
            assert curr_closedness.shape[-1] == 1
            assert curr_closedness.dim() == 4
            # Stack the num grippers dimension into the history dimension
            curr_closedness = einops.rearrange(
                curr_closedness, "b nhist ngrippers c -> b (nhist ngrippers) c"
            )
            gripper_feats = self.curr_open_close_encoder(
                curr_closedness[:, :, 0]
            )  # Remove last 1 dimension
            gripper_feats = einops.rearrange(
                gripper_feats,
                "b (nhist ngrippers c) -> b (nhist ngrippers) c",
                nhist=nhist,
                ngrippers=ngrippers,
            )
        else:
            gripper_feats = gripper_embed.weight.unsqueeze(0).repeat(len(gripper), 1, 1)

        # Rotary positional encoding
        gripper_pos = self.relative_pe_layer(
            einops.rearrange(gripper[..., :3], "B N ngrippers d -> B (N ngrippers) d")
        )
        context_pos = self.relative_pe_layer(context)

        gripper_feats = einops.rearrange(gripper_feats, "b npt c -> npt b c")
        context_feats = einops.rearrange(context_feats, "b npt c -> npt b c")
        gripper_feats, weights = self.gripper_context_head(
            query=gripper_feats, value=context_feats, query_pos=gripper_pos, value_pos=context_pos
        )
        gripper_feats = gripper_feats[-1]
        weights = weights[-1]
        gripper_feats = einops.rearrange(gripper_feats, "npt b c -> b npt c")

        return gripper_feats, gripper_pos, weights

    def encode_images(self, rgb, positions=None, valid_mask=None):
        """
        Compute visual feature embeddings at different scales.
        Additionally, if 'positions' is provided,  also compute the linearly interpolated positions
        based on the rgb features heigth and width per level.

        Args:
            - rgb: (B, ncam, 3, H, W), pixel intensities
            - positions: (B, ncam, 3, H, W), positions

        Returns:
            - rgb_feats_encoded: (B, ncam, F, feat_H_i, feat_W_i)
            - positions_encoded (Optional): (B, ncam * feat_H_i * feat_W_i, 3)
            - postions_valid_mask (Optional): (B, ncam * feat_H_i * feat_W_i)
        """
        num_cameras = rgb.shape[1]

        # Compute features from the image
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb_features = self.feature_extractor.compute(einops.rearrange(rgb, "b c h w -> b h w c"))
        rgb_features = einops.rearrange(rgb_features, "b h w c -> b c h w")

        # Encode features to to embedding dim
        width = rgb_features.shape[-1]
        height = rgb_features.shape[-2]
        rgb_features = self.image_feature_encoder(
            einops.rearrange(rgb_features, "b c h w -> b (h w) c")
        )
        rgb_features = einops.rearrange(rgb_features, "b (h w) c -> b c h w", h=height, w=width)
        feat_c, feat_h, feat_w = rgb_features.shape[-3:]

        # Interpolate the positions to fit the feature grid size
        positions = einops.rearrange(positions, "bt ncam c h w -> (bt ncam) c h w")
        positions = F.interpolate(positions, (feat_h, feat_w), mode="bilinear")

        # Resize the valid mask to fit the image features
        if valid_mask is not None:
            assert feat_h == feat_w, "Image must be square"
            downscale_factor = int(valid_mask.shape[-1] / feat_w)
            downscaled_mask = downscale_mask(valid_mask, downscale_factor)
        else:
            downscaled_mask = None

        # Reshape
        rgb_feats_encoded = einops.rearrange(
            rgb_features, "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )

        positions_encoded = einops.rearrange(
            positions, "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        if downscaled_mask is not None:
            downscaled_mask = einops.rearrange(downscaled_mask, "b ncam h w -> b (ncam h w)")

        return rgb_feats_encoded, positions_encoded, downscaled_mask

    def encode_feature_pointcloud(self, features, points):
        # Get either the shared encoder or the dedicated one for mesh features.
        encoder = (
            self.image_feature_encoder
            if self.use_shared_feature_encoder
            else self.reconstruction_encoder
        )

        assert encoder is not None, "No encoder found for mesh features"
        assert features.shape[-1] == encoder.in_features, (
            f"Wrong dimensionality of input features. "
            f"Expected {encoder.in_features}. have {features.shape[-1]}. Check type of features in dataset"
        )
        features = encoder(features)
        assert points.dtype == torch.float32
        return features.to(torch.float32), points

    def encode_instruction(self, instruction):
        """
        Compute language features/pos embeddings on top of CLIP features.

        Args:
            - instruction: (B, max_instruction_length, 512)

        Returns:
            - instr_feats: (B, 53, F)
            - instr_dummy_pos: (B, 53, F, 2)
        """
        instr_feats = self.instruction_encoder(instruction)
        # Dummy positional embeddings, all 0s
        instr_dummy_pos = torch.zeros(
            len(instruction), instr_feats.shape[1], 3, device=instruction.device
        )
        instr_dummy_pos = self.relative_pe_layer(instr_dummy_pos)
        return instr_feats, instr_dummy_pos

    def run_fps(self, context_features, context_pos, context_valid_mask):
        """
        Subsample using furthest point sampling

        Number of sampled points: Nps = Np/fps_subsampling_factor

        Args:
            context_features: (Np, B, F): feature to sample
            context_pos:      (B, Np, F, 2)
            context_valid_mask:     (B, Np)

        Returns:
            - sampled_context_features: (Nps, B, F): feature to sample
            - sampled_context_pos:      (B, Nps, F, 2)
            - sampled_context_mask:     (B, Nps)
        """

        npts, b, ch = context_features.shape

        # The FPS algorithm require consistent batch sizes. We can therefore not single out the
        # valid entries in advance, since the number of valid entries may vary across batches.
        # Instead, we indicate invalid entries by zeroing them. At the end, we return a mask that indicates zeroed items.
        # Normally, we'd expect at max one zeroed item, i.e we "lose" one feature. We can live with that.
        # If we didn't reach the desired num elements there might be several.
        context_features_masked = context_features.clone()
        context_features_masked[~context_valid_mask.transpose(0, 1)] = 0

        # Sample points with FPS
        sampled_inds = dgl_geo.farthest_point_sampler(
            einops.rearrange(context_features_masked, "npts b c -> b npts c"),
            max(npts // self.fps_subsampling_factor, 1),
            0,
        ).long()

        # Sample features
        expanded_sampled_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_context_features = torch.gather(
            context_features_masked,
            0,
            einops.rearrange(expanded_sampled_inds, "b npts c -> npts b c"),
        )

        # Sample positional embeddings
        _, _, ch, npos = context_pos.shape
        expanded_sampled_inds = sampled_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ch, npos)
        sampled_context_pos = torch.gather(context_pos, 1, expanded_sampled_inds)

        # Mask is valid where there's at least one non-zero element.
        num_sampled_points = sampled_inds.shape[-1]
        sampled_valid_mask = torch.zeros(
            size=(b, num_sampled_points), dtype=torch.bool, device=context_valid_mask.device
        )
        sampled_valid_mask[torch.any(sampled_context_features != 0, dim=-1).transpose(0, 1)] = True

        assert torch.all(
            sampled_context_features[~sampled_valid_mask.transpose(0, 1)] == 0
        ), "Features should be zero whenever the mask is inactive"

        # Check that active samples have at least one non-zero feature, but don't check samples that are all masked out.
        active_samples = sampled_valid_mask.any(dim=1)
        if torch.any(active_samples):
            assert torch.any(
                sampled_context_features[:, active_samples, :][
                    sampled_valid_mask[active_samples].transpose(0, 1)
                ]
                != 0
            ), "At least one feature should be non-zero whenever the mask is active (as long as the whole sample is not inactive)"

        return sampled_context_features, sampled_context_pos, sampled_valid_mask

    def vision_language_attention(self, feats, instr_feats):
        feats, _ = self.vl_attention[0](
            seq1=feats,
            seq1_key_padding_mask=None,
            seq2=instr_feats,
            seq2_key_padding_mask=None,
            seq1_pos=None,
            seq2_pos=None,
            seq1_sem_pos=None,
            seq2_sem_pos=None,
        )
        return feats

import einops
import torch
import torch.nn as nn

from mindmap.diffuser_actor.layers import (
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfAttentionModule,
    FFWRelativeSelfCrossAttentionModule,
    ParallelAttention,
)
from mindmap.diffuser_actor.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb


class DiffusionHead(nn.Module):
    def __init__(
        self,
        embedding_dim=60,
        num_attn_heads=8,
        use_instruction=False,
        rotation_parametrization="quat",
        nhist=3,
        prediction_horizon=1,
        ngrippers=1,
        lang_enhanced=False,
        predict_head_yaw=False,
        diffusion_dropout=0.0,
        predictor_dropout=0.0,
    ):
        super().__init__()
        self.use_instruction = use_instruction
        self.lang_enhanced = lang_enhanced
        self.ngrippers = ngrippers
        self.prediction_horizon = prediction_horizon
        self.embedding_dim = embedding_dim
        self.predict_head_yaw = predict_head_yaw
        assert "6D" in rotation_parametrization
        if "6D" in rotation_parametrization:
            rotation_dim = 6  # continuous 6D
        else:
            rotation_dim = 4  # quaternion

        # Encoders
        self.traj_encoder = nn.Sequential(
            nn.Linear(9, embedding_dim), nn.Dropout(diffusion_dropout)
        )
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(diffusion_dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.gripper_history_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist * ngrippers, embedding_dim),
            nn.ReLU(),
            nn.Dropout(diffusion_dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = nn.ModuleList(
            [
                ParallelAttention(
                    num_layers=1,
                    d_model=embedding_dim,
                    n_heads=num_attn_heads,
                    dropout=diffusion_dropout,
                    self_attention1=False,
                    self_attention2=False,
                    cross_attention1=True,
                    cross_attention2=False,
                    rotary_pe=False,
                    apply_ffn=False,
                )
            ]
        )

        # Estimate attends to context (no subsampling)
        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, dropout=diffusion_dropout, use_adaln=True
        )

        # Shared attention layers
        if not self.lang_enhanced:
            self.self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim,
                num_attn_heads,
                num_layers=4,
                dropout=diffusion_dropout,
                use_adaln=True,
            )
        else:  # interleave cross-attention to language
            self.self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim,
                num_attn_heads,
                num_self_attn_layers=4,
                num_cross_attn_layers=3,
                dropout=diffusion_dropout,
                use_adaln=True,
            )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.Dropout(diffusion_dropout)
        )
        if not self.lang_enhanced:
            self.rotation_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, dropout=diffusion_dropout, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.rotation_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, dropout=diffusion_dropout, use_adaln=True
            )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(embedding_dim, rotation_dim),
        )

        # 2. Position
        self.position_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.Dropout(diffusion_dropout)
        )
        if not self.lang_enhanced:
            self.position_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, dropout=diffusion_dropout, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.position_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, dropout=diffusion_dropout, use_adaln=True
            )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(embedding_dim, 3),
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(embedding_dim, 1),
        )

        # 4. Head yaw
        self.head_yaw_predictor = None
        if self.predict_head_yaw:
            self.head_yaw_predictor = nn.Sequential(
                nn.Linear(embedding_dim * ngrippers, embedding_dim),
                nn.ReLU(),
                nn.Dropout(predictor_dropout),
                nn.Linear(embedding_dim, 1),
            )

    def forward(
        self,
        trajectory,
        timestep,
        context_feats,
        context,
        context_mask,
        instr_feats,
        adaln_gripper_feats,
        fps_feats,
        fps_pos,
        fps_mask,
    ):
        """
        Arguments:
            trajectory: (B, trajectory_length, ngrippers, 3+6+X)
            timestep: (B, 1)
            context_feats: (B, N, F)
            context: (B, N, F, 2)
            instr_feats: (B, max_instruction_length, F)
            adaln_gripper_feats: (B, nhist, F)
            fps_feats: (N, B, F), N < context_feats.size(1)
            fps_pos: (B, N, F, 2)
        """
        # Compute embeddings of the gt-trajectory
        assert trajectory.shape[-1] == 9
        num_grippers = trajectory.shape[2]
        traj_feats = self.traj_encoder(trajectory)  # (B, L, ngrippers, F)

        # Join the gripper dimension with the trajectory length dimension
        # We do this so the position time embedding is different for each gripper
        traj_feats = einops.rearrange(traj_feats, "b l ngrippers c -> b (l ngrippers) c")

        # Trajectory features cross-attend to context features
        # Add positional embedding to the trajectory token about to undergo diffusion.
        # This is critical to allowing the trajectory to decern amongst the timestamps/grippers
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        if self.use_instruction:
            traj_feats, _ = self.traj_lang_attention[0](
                seq1=traj_feats,
                seq1_key_padding_mask=None,
                seq2=instr_feats,
                seq2_key_padding_mask=None,
                seq1_pos=None,
                seq2_pos=None,
                seq1_sem_pos=traj_time_pos,
                seq2_sem_pos=None,
            )

        # We joint the gripper dimension with the trajectory length dimension
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        traj_feats = einops.rearrange(traj_feats, "b l c -> l b c")
        context_feats = einops.rearrange(context_feats, "b l c -> l b c")
        adaln_gripper_feats = einops.rearrange(adaln_gripper_feats, "b l c -> l b c")
        pos_pred, rot_pred, openess_pred, head_yaw_pred, cross_attn_weights = self.prediction_head(
            trajectory[..., :3],
            traj_feats,
            context[..., :3],
            context_feats,
            context_mask,
            timestep,
            adaln_gripper_feats,
            fps_feats,
            fps_pos,
            fps_mask,
            instr_feats,
        )
        # We split the gripper dimension back out
        pos_pred = einops.rearrange(
            pos_pred, "b (l ngrippers) c -> b l ngrippers c", ngrippers=num_grippers
        )
        rot_pred = einops.rearrange(
            rot_pred, "b (l ngrippers) c -> b l ngrippers c", ngrippers=num_grippers
        )
        openess_pred = einops.rearrange(
            openess_pred, "b (l ngrippers) c -> b l ngrippers c", ngrippers=num_grippers
        )
        return (
            [torch.cat((pos_pred, rot_pred, openess_pred), -1)],
            head_yaw_pred,
            cross_attn_weights,
        )

    def prediction_head(
        self,
        gripper_pcd,
        gripper_features,
        context_pcd,
        context_features,
        context_mask,
        timesteps,
        gripper_history_features,
        sampled_context_features,
        sampled_rel_context_pos,
        sampled_context_mask,
        instr_feats,
    ):
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, F)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, F)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            gripper_history_features: A tensor of shape (M, B, F)
            sampled_context_features: A tensor of shape (K, B, F)
            sampled_rel_context_pos: A tensor of shape (B, K, F, 2)
            instr_feats: (B, max_instruction_length, F)
        """

        # Handle samples where all points are masked out. This might happen if we're running RGBD-only with a large depth threshold or with a camera pointing outside the bounding box.
        # Best we can do is to set the mask to all-active to avoid nan attention weights. We also zero the relevant features to reduce their impact.
        empty_samples = ~context_mask.any(dim=-1)
        empty_samples_fps = ~sampled_context_mask.any(dim=-1)
        context_mask[empty_samples] = True
        sampled_context_mask[empty_samples_fps] = True
        context_features[:, empty_samples] = 0
        sampled_context_features[:, empty_samples_fps] = 0

        if torch.any(empty_samples) or torch.any(empty_samples_fps):
            print(
                f"Warning: {torch.sum(empty_samples)} samples and {torch.sum(empty_samples_fps)} fps samples have all points masked out. Setting mask to all-active to avoid nan attention weights."
            )

        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(timesteps, gripper_history_features)

        # Positional embeddings
        rel_gripper_pos = self.relative_pe_layer(
            einops.rearrange(gripper_pcd, "B N ngrippers d -> B (N ngrippers) d")
        )
        rel_context_pos = self.relative_pe_layer(context_pcd)
        # Cross attention from gripper to full context
        gripper_features, cross_attn_weights = self.cross_attn(
            query=gripper_features,
            value=context_features,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs,
            key_padding_mask=~context_mask,  # Need invert since key_padding_mask is an exclusion mask
        )
        gripper_features = gripper_features[-1]

        # Self attention among gripper and sampled context
        features = torch.cat([gripper_features, sampled_context_features], 0)
        rel_pos = torch.cat([rel_gripper_pos, sampled_rel_context_pos], 1)

        # Resize the mask to accomodate the gripper. We add one batch-sized column of "False" values since the gripper is never deactivated
        num_gripper_preds = gripper_features.shape[0]
        assert num_gripper_preds == self.ngrippers * self.prediction_horizon
        batch_size = context_mask.shape[0]
        combined_mask = torch.cat(
            [
                torch.zeros(
                    (batch_size, num_gripper_preds), dtype=torch.bool, device=context_mask.device
                ),
                ~sampled_context_mask,
            ],
            dim=1,
        )

        assert torch.all(combined_mask[:, 0]) == False, "Gripper should never be deactivated"

        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None,
            key_padding_mask=combined_mask.squeeze(-1),
        )
        features = features[-1]

        # Rotation head
        rotation = self.predict_rot(
            features, rel_pos, time_embs, num_gripper_preds, instr_feats, combined_mask
        )

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper_preds, instr_feats, combined_mask
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        # Head yaw from position head
        # NOTE(remos): We use the position features of all grippers to predict the head yaw.
        assert position_features.ndim == 3
        assert position_features.shape[1] == num_gripper_preds
        assert position_features.shape[2] == self.embedding_dim
        multi_gripper_position_features = einops.rearrange(
            position_features, "b (l ngrippers) c -> b l (ngrippers c)", ngrippers=self.ngrippers
        )
        head_yaw = None
        if self.head_yaw_predictor is not None:
            head_yaw = self.head_yaw_predictor(multi_gripper_position_features)

        # Average cross attention weights over all attention heads
        cross_attn_weights = torch.mean(cross_attn_weights[-1], dim=1).squeeze(1)

        # Make sure there are no nan values in the output. This might be the case if a sample is completely masked out, which should normally be handled properly.
        assert not torch.any(cross_attn_weights[-1].isnan())
        assert not torch.any(gripper_features.isnan())
        assert not torch.any(features.isnan())
        assert not torch.any(position.isnan())
        assert not torch.any(rotation.isnan())
        assert not torch.any(openess.isnan())

        if head_yaw is not None:
            assert not torch.any(head_yaw.isnan())

        return position, rotation, openess, head_yaw, cross_attn_weights

    def encode_denoising_timestep(self, timestep, gripper_history_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)

        gripper_history_features = einops.rearrange(
            gripper_history_features, "npts b c -> b npts c"
        )
        gripper_history_features = gripper_history_features.flatten(1)
        gripper_history_feats = self.gripper_history_emb(gripper_history_features)
        return time_feats + gripper_history_feats

    def predict_pos(self, features, rel_pos, time_embs, num_gripper, instr_feats, exclusion_mask):
        position_features = self.position_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None,
            key_padding_mask=exclusion_mask,
        )
        position_features = position_features[-1]
        position_features = einops.rearrange(
            position_features[:num_gripper], "npts b c -> b npts c"
        )
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, rel_pos, time_embs, num_gripper, instr_feats, exclusion_mask):
        rotation_features = self.rotation_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None,
            key_padding_mask=exclusion_mask,
        )
        rotation_features = rotation_features[-1]
        rotation_features = einops.rearrange(
            rotation_features[:num_gripper], "npts b c -> b npts c"
        )
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation

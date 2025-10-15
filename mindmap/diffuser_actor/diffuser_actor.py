from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import einops
from nvblox_torch.timer import Timer
import torch
import torch.nn as nn

from mindmap.data_loading.data_types import DataType, includes_mesh, includes_pcd
from mindmap.diffuser_actor.diffusion_head import DiffusionHead
from mindmap.diffuser_actor.encoder import Encoder
from mindmap.image_processing.feature_extraction import FeatureExtractorType
from mindmap.model_utils.distributed_training import get_rank
from mindmap.model_utils.loss import LossWeights, compute_loss
from mindmap.model_utils.normalization import (
    normalize_pointcloud,
    normalize_pos,
    normalize_trajectory,
    unnormalize_trajectory,
)
from mindmap.model_utils.relative_conversions import (
    get_current_pose_from_gripper_history,
    to_absolute_trajectory,
    to_relative_gripper_history,
    to_relative_pcd,
    to_relative_trajectory,
)
from mindmap.visualization.tensor_visualizer import TensorVisualizer


class DiffuserActor(nn.Module):
    def __init__(
        self,
        feature_type=FeatureExtractorType.CLIP_RESNET50_FPN,
        image_size=(256, 256),
        feature_image_size=(32, 32),
        embedding_dim=60,
        num_vis_ins_attn_layers=2,
        use_instruction=False,
        fps_subsampling_factor=5,
        workspace_bounds=None,
        rotation_parametrization="6D",
        quaternion_format="xyzw",
        diffusion_timesteps=100,
        nhist=3,
        ngrippers=1,
        prediction_horizon=1,
        relative=False,
        lang_enhanced=False,
        predict_head_yaw=False,
        data_type=DataType.RGBD,
        use_fps=True,
        encode_openness=False,
        use_shared_feature_encoder=False,
        loss_weights: LossWeights = LossWeights(),
        encoder_dropout=0.0,
        diffusion_dropout=0.0,
        predictor_dropout=0.0,
        add_external_cam=True,
        fpn_path=None,
    ):
        """Constructor.

        Args:
          feature_type:                 Feature extractor type.
          image_size:                   Input image size. One of [(128, 128), (256, 256), (512, 512)].
          feature_image_size:           The desired dimension of the feature extractor output.
          embedding_dim:                Dimension of context and gripper features.
          num_vis_ins_attn_layers:      Num layers in the encoder's vision<->language transformer.
          use_instruction:              Whether to use language instruction in the diffusion head.
          fps_subsampling_factor:       Subsampling used for Furthest Point Sampling.
          workspace_bounds:             Bounds used for normalization of 3D input data.
          rotation_parametrization:     Internal rotation representation. One of ["6D", "quat"].
          quaternion_format:            Format of input pose. One of ["xyzw", "wxyz"].
          diffusion_timesteps:          Number of de-noising steps.
          nhist:                        Number of previous gripper poses to as input.
          ngrippers:                    Number of grippers to predict.
          prediction_horizon:           Number of future poses to predict.
          relative:                     Whether to estimate poses relative to the most recent pose.
          lang_enhanced:                Whether to apply vision<->language cross attention in the diffusion head.
          predict_head_yaw:             Whether to predict head yaw in the diffusion head.
          data_type:                    Type of input data.
          use_fps:                      Whether to apply furthest point sampling on the encoded features.
          encode_openness:              Whether to encode gripper openness in the history.
          use_shared_feature_encoder:   Whether to use shared feature encoder.
          loss_weights:                 Loss weights for different components.
          encoder_dropout:              Dropout rate for the encoder.
          diffusion_dropout:            Dropout rate for the diffusion head.
          predictor_dropout:            Dropout rate for the prediction heads.
          add_external_cam:             Whether to add the external cam data.
          fpn_path:                     Path to pre-trained feature pyramid network (only used for CLIP features).
        """
        super().__init__()
        self._rotation_parametrization = rotation_parametrization
        self._quaternion_format = quaternion_format
        self._relative = relative
        self.use_instruction = use_instruction
        self.pca_stats = None
        self._use_fps = use_fps
        self.loss_weights = loss_weights
        self._data_type = data_type
        self._add_external_cam = add_external_cam
        self._num_cams = 2 if self._add_external_cam else 1
        self._nhist = nhist
        self._prediction_horizon = prediction_horizon
        self._predict_head_yaw = predict_head_yaw

        if get_rank() == 0:
            self.vis = TensorVisualizer()
            if self._data_type == DataType.RGBD:
                if self._add_external_cam:
                    self.vis.register_tensor("image_rgb", (2, 3, 256, 256), nrow=1)
                    self.vis.register_tensor("context_feat_table_cam", (120, 1, 32, 32), nrow=30)
                    self.vis.register_tensor("context_feat_wrist_cam", (120, 1, 32, 32), nrow=30)
                    # attn mask for gripper history: (nhist: gripper_history, 2:cams, 8:gripper_dim)
                    self.vis.register_tensor(
                        "encode_gripper_history_feats_attn_mask",
                        (nhist * 2 * 8, 1, 32, 32),
                        nrow=12,
                    )

        self.encoder = Encoder(
            feature_type=feature_type,
            image_size=image_size,
            feature_image_size=feature_image_size,
            embedding_dim=embedding_dim,
            nhist=nhist,
            ngrippers=ngrippers,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            data_type=data_type,
            encode_openness=encode_openness,
            encoder_dropout=encoder_dropout,
            fpn_path=fpn_path,
            use_shared_feature_encoder=use_shared_feature_encoder,
        )
        self.prediction_head = DiffusionHead(
            embedding_dim=embedding_dim,
            use_instruction=use_instruction,
            rotation_parametrization=rotation_parametrization,
            nhist=nhist,
            prediction_horizon=prediction_horizon,
            ngrippers=ngrippers,
            lang_enhanced=lang_enhanced,
            predict_head_yaw=predict_head_yaw,
            diffusion_dropout=diffusion_dropout,
            predictor_dropout=predictor_dropout,
        )
        self.position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )
        self.rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        self.n_steps = diffusion_timesteps
        self.workspace_bounds = workspace_bounds

    def encode_inputs(
        self,
        visible_rgb,
        visible_pcd,
        visible_pcd_valid_mask,
        vertex_features,
        vertices,
        vertices_valid_mask,
        instruction,
        gripper_history,
        curr_closedness,
    ):
        """
        Encode vision, gripper and language inputs

        Args:
           visible_rgb:  (B, ncam, 3, H, W), Pixel intensities.
           visible_pcs:  (B, ncam, 3, H, W), Point cloud.
           vertex_features: (B, num_vertices, nvblox_feature_dim)
           vertices:        (B, num_vertices, 3)
           vertices_valid_mask: (B, num_vertices)
           instruction:  (B, max_instruction_length, 512), Language instruction.
           gripper_history: (B, nhist, ngrippers, 9), Gripper pose history.

        Returns:
           context_feats:       (B, 2048, ncam*embedding_dim), Features describing the context.
           context:             (B, 2048, 3), The context, e.g. 3D points.
           instr_features:      (?), Features extracted from the instruction - attended to context.
           adaln_gripper_feats: (B, 3, ncam*embedding_dim), Gripper features.
           fps_feats:           (2048/subsampl, 16, ncam*embedding_dim), Subsampled features.
           fps_pos:             (B, 2048/subsampl, ncam*embedding_dim, 2), Subsampled positions.
        """

        if vertices is not None:
            assert vertices.ndim == 3, f"Expected [b, n_vertices, 3]. Have: f{vertices.ndim}"
        if vertices_valid_mask is not None:
            assert (
                vertices_valid_mask.ndim == 2
            ), f"Expexcted [b, num_vertices]. Have: {vertices_valid_mask.ndim}"
            assert vertices is not None and vertices.shape[1] == vertices_valid_mask.shape[1]
        if visible_pcd_valid_mask is not None:
            assert (
                visible_pcd_valid_mask.ndim == 4
            ), f"Expected [b, ncam, w, h]. Have: {visible_pcd_valid_mask.ndim}"

        if self._data_type == DataType.RGBD:
            with Timer("diffuser_actor/encode_inputs/images"):
                # Compute visual features/positional embeddings at different scales
                context_feats, context, context_mask = self.encoder.encode_images(
                    visible_rgb, visible_pcd, valid_mask=visible_pcd_valid_mask
                )

                if (
                    get_rank() == 0
                    and self.vis is not None
                    and not self.vis.enabled
                    and self._add_external_cam
                ):
                    for i in range(context_feats.shape[1]):
                        self.vis.register_tensor(f"context_feat_cam_{i}", (120, 1, 32, 32), nrow=30)
                        self.vis.set(f"context_feat_cam_{i}", context_feats[0, i].unsqueeze(1))

        elif self._data_type == DataType.MESH:
            assert self.encoder.reconstruction_encoder is not None
            context_feats, context = self.encoder.encode_feature_pointcloud(
                vertex_features, vertices
            )
            context_mask = vertices_valid_mask

        elif self._data_type == DataType.RGBD_AND_MESH:
            wrist_context_feats, wrist_context, wrist_context_mask = self.encoder.encode_images(
                visible_rgb, visible_pcd, valid_mask=visible_pcd_valid_mask
            )
            mesh_context_feats, mesh_context = self.encoder.encode_feature_pointcloud(
                vertex_features, vertices
            )

            # Concatenate wrist and mesh context features and masks
            context_feats = torch.cat([wrist_context_feats, mesh_context_feats], dim=1)
            context = torch.cat([wrist_context, mesh_context], dim=1)
            context_mask = torch.cat([wrist_context_mask, vertices_valid_mask], dim=1)
        else:
            raise NotImplementedError(f"Data type not implemented: {self._data_type}")

        # Encode instruction (B, 53, F)
        instr_feats = None
        if self.use_instruction:
            with Timer("diffuser_actor/encode_inputs/instruction"):
                print("Using language instruction.")
                instr_feats, _ = self.encoder.encode_instruction(instruction)
                # Cross-attention vision to language
                context_feats = self.encoder.vision_language_attention(context_feats, instr_feats)

        # Encode gripper history (B, nhist, ngrippers, F)
        with Timer("diffuser_actor/encode_inputs/gripper_history"):
            adaln_gripper_feats, _, weights = self.encoder.encode_gripper_history(
                gripper_history, context_feats, context, curr_closedness
            )
            if (
                get_rank() == 0
                and self.vis is not None
                and self.vis.enabled
                and self._add_external_cam
            ):
                if self._data_type == DataType.RGBD:
                    ncams = visible_rgb.shape[1]
                    spatial_dim = int(torch.sqrt(torch.tensor(context_feats.shape[1] / ncams)))
                    attn_mask = einops.rearrange(
                        weights[0],
                        "c nhist (ncam h w) -> (c nhist ncam) h w",
                        ncam=ncams,
                        h=spatial_dim,
                        w=spatial_dim,
                    )
                else:
                    attn_mask_numel = weights[0].shape[-1]
                    attn_mask_spatial_dim = 16
                    attn_mask = einops.rearrange(
                        weights[0],
                        "c nhist (h w d) -> (c nhist d) h w",
                        d=int(
                            round(attn_mask_numel / (attn_mask_spatial_dim * attn_mask_spatial_dim))
                        ),
                        h=attn_mask_spatial_dim,
                        w=attn_mask_spatial_dim,
                    )
                self.vis.set(
                    "encode_gripper_history_feats_attn_mask",
                    attn_mask.unsqueeze(1),
                    value_range=(attn_mask.min(), attn_mask.max()),
                )

        # FPS on visual features (N, B, F) and (B, N, F, 2)
        with Timer("diffuser_actor/encode_inputs/fps"):
            if self._use_fps:
                fps_feats, fps_pos, fps_mask = self.encoder.run_fps(
                    context_feats.transpose(0, 1),
                    self.encoder.relative_pe_layer(context),
                    context_mask,
                )
            else:
                fps_feats = context_feats.transpose(0, 1)
                fps_pos = self.encoder.relative_pe_layer(context)
                fps_mask = context_mask
        return {
            "context_feats": context_feats,
            "context": context,  # contextualized visual features
            "context_mask": context_mask.squeeze(-1),  # mask for contextualized visual features
            "instr_feats": instr_feats,  # language features
            "adaln_gripper_feats": adaln_gripper_feats,  # gripper history features
            "fps_feats": fps_feats,
            "fps_pos": fps_pos,  # sampled visual features
            "fps_mask": fps_mask,  # mask for sampled visual features
        }

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        """
        De-noise trajectory.

        Args:
          trajectory:   (B, npred, ngrippers, 9), Noisy trajectory.
          timestep:     (B), De-nosing timesteps to run.
          fixed_inputs:  Encoded input data, i.e. output from encode_inputs().

        Returns:
          De-noised trajectory: [(B, npred, ngrippers, 10)], pos + rot + gripper_open
        """

        return self.prediction_head(
            trajectory,
            timestep,
            context_feats=fixed_inputs["context_feats"],
            context=fixed_inputs["context"],
            context_mask=fixed_inputs["context_mask"],
            instr_feats=fixed_inputs["instr_feats"],
            adaln_gripper_feats=fixed_inputs["adaln_gripper_feats"],
            fps_feats=fixed_inputs["fps_feats"],
            fps_pos=fixed_inputs["fps_pos"],
            fps_mask=fixed_inputs["fps_mask"],
        )

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs):
        """
        Predict a trajectory using conditional input.

        Args:
          condition_data: (B, npred, 9)
          condition_mask: (B, npred, 9)
          fixed_inputs:   Encoded input data, i.e. output from encode_inputs().

        Return:
          trajectory: (B, npred, 10), pos + rot + gripper_open
        """
        self.position_noise_scheduler.set_timesteps(self.n_steps)
        self.rotation_noise_scheduler.set_timesteps(self.n_steps)

        # Random trajectory, conditioned on start-end
        noise = torch.randn(
            size=condition_data.shape, dtype=condition_data.dtype, device=condition_data.device
        )
        # Noisy condition data
        noise_t = (
            torch.ones((len(condition_data),), device=condition_data.device)
            .long()
            .mul(self.position_noise_scheduler.timesteps[0])
        )
        noise_pos = self.position_noise_scheduler.add_noise(
            condition_data[..., :3], noise[..., :3], noise_t
        )
        noise_rot = self.rotation_noise_scheduler.add_noise(
            condition_data[..., 3:9], noise[..., 3:9], noise_t
        )
        noisy_condition_data = torch.cat((noise_pos, noise_rot), -1)
        trajectory = torch.where(condition_mask, noisy_condition_data, noise)

        # Iterative denoising
        timesteps = self.position_noise_scheduler.timesteps
        # timesteps torch tensor: range(100, -1, -1)
        assert trajectory.shape[1] > 0, "Must have more than one point in trajectory"
        cross_attn_weights_acc = None
        for t in timesteps:
            traj_pred, head_yaw_pred, cross_attn_weights = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(trajectory.device).long(),
                fixed_inputs,
            )
            if cross_attn_weights_acc is None:
                cross_attn_weights_acc = cross_attn_weights
            else:
                cross_attn_weights_acc += cross_attn_weights
            traj_pred = traj_pred[-1]  # keep only last layer's output
            assert traj_pred.shape[-1] == 9 + 1
            pos = self.position_noise_scheduler.step(
                traj_pred[..., :3], t, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_noise_scheduler.step(
                traj_pred[..., 3:9], t, trajectory[..., 3:9]
            ).prev_sample
            trajectory = torch.cat((pos, rot), -1)

        openess_pred = traj_pred[..., 9:]
        trajectory = torch.cat((trajectory, openess_pred), -1)

        # NOTE(remos): openess and head_yaw are not going through diffusion steps.
        return trajectory, head_yaw_pred, cross_attn_weights / len(timesteps)

    def gt_provided(self, gt_gripper_pred, gt_openness, gt_head_yaw):
        """Check if ground-truth data is provided for gripper, openness, and optionally head yaw."""
        if self._predict_head_yaw:
            return (
                gt_gripper_pred is not None and gt_openness is not None and gt_head_yaw is not None
            )
        else:
            return gt_gripper_pred is not None and gt_openness is not None

    def compute_trajectory(
        self,
        gt_gripper_pred,
        gt_openness,
        gt_head_yaw,
        rgb_obs,
        pcd_obs,
        pcd_valid_mask,
        vertex_features,
        vertices,
        vertices_valid_mask,
        instruction,
        gripper_history,
        current_pose,
        current_openness,
    ):
        """
        Predict npred future poses.

        Args:
          gt_gripper_pred:   (B, npred, ngrippers, 9),  Optional ground-truth trajectory for computing losses.
          gt_openess:      (B, npred, ngrippers, 1),  Optional ground-truth gripper open/close states.
          gt_head_yaw:     (B, npred, 1),  Optional ground-truth head yaw.
          rgb_obs:         (B, ncams, 3, H, W), RGB observations.
          pcd_obs:         (B, ncams, 3, H, W), Depth points.
          vertex_features: (B, num_vertices, nvblox_feature_dim)
          vertices:        (B, num_vertices, 3)
          vertices_valid_mask: (B, num_vertices)
          instruction:     (?), Natural language instruction.
          gripper_history:    (B, nhist, ngrippers, 9), Gripper history.
          current_pose:    (?), Current pose, used only in relative pose mode.

        Returns:
          trajectory_pred: (B, npred, ngrippers, 8), Predicted poses.
          head_yaw_pred:   (B, npred, 1), Predicted head yaw.
          losses:          (npred), Losses. None if no gt was proveded.
          encoded_inputs:  (dict), Dict containing encoded inputs
        """
        # NOTE(remos): We expect normalized data (and rotation in 6D form).
        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs,
            pcd_obs,
            pcd_valid_mask,
            vertex_features,
            vertices,
            vertices_valid_mask,
            instruction,
            gripper_history,
            current_openness,
        )

        # Condition on start-end pose
        B, nhist, ngrippers, D = gripper_history.shape
        assert self._nhist == nhist
        cond_data = torch.zeros(
            (B, self._prediction_horizon, ngrippers, D), device=gripper_history.device
        )
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
        trajectory_pred, head_yaw_pred, cross_attn_weights = self.conditional_sample(
            cond_data, cond_mask, fixed_inputs
        )

        # Compute loss if gt is provided.
        losses = None
        if self.gt_provided(gt_gripper_pred, gt_openness, gt_head_yaw):
            assert self._prediction_horizon == gt_gripper_pred.shape[1]
            assert self._prediction_horizon == gt_openness.shape[1]
            assert (
                self._prediction_horizon == gt_head_yaw.shape[1] if self._predict_head_yaw else True
            )
            losses = compute_loss(
                trajectory_pred,
                head_yaw_pred,
                gt_gripper_pred,
                gt_openness,
                gt_head_yaw,
                loss_weights=self.loss_weights,
                predict_head_yaw=self._predict_head_yaw,
                rotation_form="6D",
            )

        # Unnormalize data.
        trajectory_pred = unnormalize_trajectory(
            trajectory_pred,
            self.workspace_bounds,
            self._rotation_parametrization,
            self._quaternion_format,
        )

        # Convert back to absolute pose.
        if self._relative:
            trajectory_pred = to_absolute_trajectory(trajectory_pred, current_pose)

        # Clamp head_yaw_pred to [-pi, pi)
        if self._predict_head_yaw:
            head_yaw_pred = torch.clamp(head_yaw_pred, min=-torch.pi, max=torch.pi - 1e-6)

        return trajectory_pred, head_yaw_pred, losses, fixed_inputs, cross_attn_weights

    def forward(
        self,
        gt_gripper_pred,
        gt_head_yaw,
        rgb_obs,
        pcd_obs,
        pcd_valid_mask,
        vertex_features,
        vertices,
        vertices_valid_mask,
        instruction,
        gripper_history,
        run_inference=False,
    ):
        """
        Arguments:
            gt_gripper_pred: (B, prediction_horizon, 3+4+X)
            gt_head_yaw: (B, prediction_horizon, 1)
            rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
            pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
            vertex_features: (B, num_vertices, nvblox_feature_dim)
            vertices: (B, num_vertices, 3)
            instruction: (B, max_instruction_length, 512)
            gripper_history: (B, nhist, 3+4+X)

        Note:
            Regardless of rotation parametrization, the input rotation
            is ALWAYS expressed as a quaternion form.
            The model converts it to 6D internally if needed.
        """
        # Extract the gripper open/closed from the history
        curr_closedness = torch.unsqueeze(gripper_history[..., -1], 3)

        # Note (xyao): gripper_history only takes pos + quat
        gripper_history = gripper_history[..., :7]

        # Convert to relative.
        current_pose = None
        if self._relative:
            # Convert pcd and gripper history relative to current gripper pose.
            current_pose = get_current_pose_from_gripper_history(gripper_history)
            # rgbd mode only
            if pcd_obs is not None:
                assert self._data_type == DataType.RGBD
                pcd_obs = to_relative_pcd(pcd_obs, current_pose)
            gripper_history = to_relative_gripper_history(gripper_history, current_pose)
            if gt_gripper_pred is not None:
                # Convert the trajectory relative to current gripper pose.
                gt_gripper_pred = to_relative_trajectory(gt_gripper_pred, current_pose)

        # Normalize data.
        #  - Gripper Positions and point cloud are normalized to the gripper location bounds
        #  - Rotations are converted to the given parameterization
        with Timer("diffuser_actor/normalize_trajectory"):
            gripper_history = normalize_trajectory(
                gripper_history,
                self.workspace_bounds,
                self._rotation_parametrization,
                self._quaternion_format,
            )
        if pcd_obs is not None:
            assert includes_pcd(self._data_type)
            # Normalize pointcloud to the gripper bounds. Points outside are excluded.
            pcd_obs, in_bounds_mask = normalize_pointcloud(pcd_obs, self.workspace_bounds)
            pcd_valid_mask &= in_bounds_mask
        if vertices is not None:
            assert includes_mesh(self._data_type)
            vertices, _ = normalize_pos(vertices, self.workspace_bounds)
        if gt_gripper_pred is not None:
            with Timer("diffuser_actor/normalize_gt_gripper_pred"):
                # Check that GT contains position, quaternion and gripper status.
                assert gt_gripper_pred.shape[-1] == 3 + 4 + 1
                gt_openess = gt_gripper_pred[..., 7:]
                gt_gripper_pred = gt_gripper_pred[..., :7]
                gt_gripper_pred = normalize_trajectory(
                    gt_gripper_pred,
                    self.workspace_bounds,
                    self._rotation_parametrization,
                    self._quaternion_format,
                )
        else:
            gt_openess = None

        if run_inference:
            with Timer("diffuser_actor/run_inference"):
                return self.compute_trajectory(
                    gt_gripper_pred,
                    gt_openess,
                    gt_head_yaw,
                    rgb_obs,
                    pcd_obs,
                    pcd_valid_mask,
                    vertex_features,
                    vertices,
                    vertices_valid_mask,
                    instruction,
                    gripper_history,
                    current_pose,
                    curr_closedness,
                )

        # TODO (xyao): why gt_gripper_pred in shape of (horizon-1, 9) -- maybe predicting the future
        assert gripper_history.shape[-1] == 9
        assert gt_gripper_pred.shape[-1] == 9

        # Prepare inputs
        with Timer("diffuser_actor/encode_inputs"):
            fixed_inputs = self.encode_inputs(
                rgb_obs,
                pcd_obs,
                pcd_valid_mask,
                vertex_features,
                vertices,
                vertices_valid_mask,
                instruction,
                gripper_history,
                curr_closedness,
            )

        with Timer("diffuser_actor/add_conditional_noisesample"):
            # Condition on start-end pose
            cond_data = torch.zeros_like(gt_gripper_pred)
            cond_mask = torch.zeros_like(cond_data)
            cond_mask = cond_mask.bool()

            # Sample noise
            noise = torch.randn(gt_gripper_pred.shape, device=gt_gripper_pred.device)

            # Sample a random timestep
            timesteps = torch.randint(
                0,
                self.position_noise_scheduler.config.num_train_timesteps,
                (len(noise),),
                device=noise.device,
            ).long()

            # Add noise to the clean trajectories
            pos = self.position_noise_scheduler.add_noise(
                gt_gripper_pred[..., :3], noise[..., :3], timesteps
            )
            rot = self.rotation_noise_scheduler.add_noise(
                gt_gripper_pred[..., 3:9], noise[..., 3:9], timesteps
            )
            noisy_trajectory = torch.cat((pos, rot), -1)
            noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition
            assert not cond_mask.any()

        # Predict the noise residual
        with Timer("diffuser_actor/policy_forward_pass"):
            trajectory_pred, head_yaw_pred, cross_attn_weights = self.policy_forward_pass(
                noisy_trajectory, timesteps, fixed_inputs
            )

        # return a list of length 1
        assert len(trajectory_pred) == 1
        trajectory_pred = trajectory_pred[0]

        # Compute loss
        with Timer("diffuser_actor/compute_loss"):
            losses = compute_loss(
                trajectory_pred,
                head_yaw_pred,
                noise,
                gt_openess,
                gt_head_yaw,
                loss_weights=self.loss_weights,
                predict_head_yaw=self._predict_head_yaw,
                rotation_form="6D",
            )
        return losses, fixed_inputs, cross_attn_weights

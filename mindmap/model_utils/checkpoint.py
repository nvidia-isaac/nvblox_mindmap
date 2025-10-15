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
import pathlib

import torch

from mindmap.cli.args import ModelArgs
from mindmap.diffuser_actor.diffuser_actor import DiffuserActor
from mindmap.embodiments.arm.embodiment import ArmEmbodiment
from mindmap.embodiments.embodiment_base import EmbodimentType
from mindmap.embodiments.humanoid.embodiment import HumanoidEmbodiment
from mindmap.embodiments.task_to_embodiment import get_embodiment_type_from_task
from mindmap.mapping.nvblox_mapper_constants import get_workspace_bounds
from mindmap.model_utils.loss import LossWeights
from mindmap.model_utils.task_to_predict_head_yaw import get_predict_head_yaw_from_task


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(checkpoint_log_dir, model, optimizer, step_id, new_loss, best_loss):
    """Save checkpoint if requested."""
    if new_loss is None or best_loss is None or new_loss <= best_loss:
        best_loss = new_loss
        torch.save(
            {
                "weight": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": step_id + 1,
                "best_loss": best_loss,
            },
            pathlib.Path(checkpoint_log_dir) / "best.pth",
        )
    torch.save(
        {
            "weight": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": step_id + 1,
            "best_loss": best_loss,
        },
        pathlib.Path(checkpoint_log_dir) / "last.pth",
    )
    return best_loss


def get_model(args: ModelArgs):
    """Initialize the model."""
    loss_weights = LossWeights(
        pos_loss=args.pos_loss, rot_loss=args.rot_loss, gripper_loss=args.gripper_loss
    )
    # Get the number of grippers from the embodiment.
    embodiment_type = get_embodiment_type_from_task(args.task)
    if embodiment_type == EmbodimentType.ARM:
        embodiment = ArmEmbodiment()
    elif embodiment_type == EmbodimentType.HUMANOID:
        embodiment = HumanoidEmbodiment(args.task)
    else:
        raise ValueError(f"Embodiment type {args.embodiment_type} not supported")

    model = DiffuserActor(
        workspace_bounds=get_workspace_bounds(args.task),
        feature_type=args.feature_type,
        image_size=args.image_size,
        feature_image_size=args.feature_image_size,
        embedding_dim=args.embedding_dim,
        num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
        use_instruction=bool(args.use_instruction),
        fps_subsampling_factor=args.fps_subsampling_factor,
        rotation_parametrization=args.rotation_parametrization,
        quaternion_format=args.quaternion_format,
        diffusion_timesteps=args.diffusion_timesteps,
        nhist=args.num_history,
        ngrippers=embodiment.get_num_grippers(),
        prediction_horizon=args.prediction_horizon,
        relative=bool(args.relative_action),
        lang_enhanced=bool(args.lang_enhanced),
        predict_head_yaw=get_predict_head_yaw_from_task(args.task),
        data_type=args.data_type,
        use_fps=args.use_fps,
        encode_openness=bool(args.encode_openness),
        use_shared_feature_encoder=bool(args.use_shared_feature_encoder),
        encoder_dropout=args.encoder_dropout,
        diffusion_dropout=args.diffusion_dropout,
        predictor_dropout=args.predictor_dropout,
        loss_weights=loss_weights,
        add_external_cam=args.add_external_cam,
        fpn_path=args.fpn_checkpoint,
    )
    print("Model parameters:", count_parameters(model))

    return model


def load_inference_checkpoint(checkpoint_path: str, model, device: str):
    assert checkpoint_path is not None
    assert os.path.exists(checkpoint_path), checkpoint_path
    print("Loading model from", checkpoint_path, flush=True)
    model_dict = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(model_dict["weight"])
    model.eval()

    model = model.to(device=device)

    return model


def load_train_checkpoint(args, model, optimizer):
    assert args.checkpoint is not None
    assert os.path.exists(args.checkpoint), args.checkpoint
    print("Loading model from", args.checkpoint, flush=True)
    model_dict = torch.load(args.checkpoint, map_location="cpu")

    model.load_state_dict(model_dict["weight"])
    if "optimizer" in model_dict:
        optimizer.load_state_dict(model_dict["optimizer"])
        for p in range(len(optimizer.param_groups)):
            optimizer.param_groups[p]["lr"] = args.initial_learning_rate
    start_iter = model_dict.get("iter", 0)
    best_loss = model_dict.get("best_loss", None)

    print(
        "=> loaded successfully '{}' (step {})".format(args.checkpoint, model_dict.get("iter", 0))
    )
    del model_dict
    torch.cuda.empty_cache()
    return start_iter, best_loss

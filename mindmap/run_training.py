# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import logging
import os
from pathlib import Path
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from datetime import datetime

import numpy as np
from nvblox_torch.timer import Timer, get_last_time, get_mean_time, timer_status_string
import plotly.graph_objs as go
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from tqdm import trange
import wandb

from mindmap.cli.args import (
    TRAINING_ARGUMENT_FILE_NAME,
    TrainingAppArgs,
    update_model_args_from_checkpoint,
)
from mindmap.common_utils.system import get_shmem_usage_mb
from mindmap.data_loading.batching import unpack_batch
from mindmap.data_loading.dataset_files_by_encoding_method import (
    get_data_loader_by_data_type,
    get_data_loader_without_augmentations,
)
from mindmap.data_loading.sampling_weighting_type import (
    SamplingWeightingType,
    get_sampling_weighting_type,
)
from mindmap.embodiments.arm.embodiment import ArmEmbodiment
from mindmap.embodiments.embodiment_base import EmbodimentType
from mindmap.embodiments.humanoid.embodiment import HumanoidEmbodiment
from mindmap.embodiments.task_to_embodiment import get_embodiment_type_from_task
from mindmap.model_utils.checkpoint import get_model, load_train_checkpoint, save_checkpoint
from mindmap.model_utils.distributed_training import (
    all_gather,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    print_dist,
)
from mindmap.model_utils.loss import compute_metrics
from mindmap.model_utils.model import print_num_trainable_params
from mindmap.model_utils.multi_gpu import MultiProcessGroup
from mindmap.model_utils.task_to_predict_head_yaw import get_predict_head_yaw_from_task
from mindmap.visualization.visualizer import Visualizer


def generate_visualizations(pred, gt):
    """Plot trajectory from the first sample in batch."""
    batch_idx = 0
    pred = pred[batch_idx].detach().cpu().numpy()
    gt = gt[batch_idx].detach().cpu().numpy()

    # Create the 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pred[:, :, 0],
                y=pred[:, :, 1],
                z=pred[:, :, 2],
                mode="markers",
                name="pred",
                marker=dict(size=5, color="red", opacity=0.8),
            ),
            go.Scatter3d(
                x=gt[:, :, 0],
                y=gt[:, :, 1],
                z=gt[:, :, 2],
                mode="markers",
                name="gt",
                marker=dict(size=5, color="blue", opacity=0.8),
            ),
        ]
    )

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"),
        title="GT(blue) vs pred(red) trajectory",
    )
    return fig


class Trainer:
    def __init__(self, args):
        """Initialize."""
        self.args = args

        # Create logging directory.
        if get_rank() == 0:
            if not os.path.exists(args.base_log_dir):
                os.makedirs(args.base_log_dir, exist_ok=True)
            # Start weight and biases.
            if args.wandb_name is not None:
                wandb_id = args.wandb_name + datetime.today().strftime("_%Y.%m.%d-%H.%M.%S")
                wandb.init(
                    entity=args.wandb_entity,
                    project=args.exp_name,
                    dir=args.base_log_dir,
                    id=wandb_id,
                    config=args,
                )
            else:
                wandb.init(
                    entity=args.wandb_entity,
                    project=args.exp_name,
                    dir=args.base_log_dir,
                    config=args,
                    mode=args.wandb_mode,
                )

            # Create directory for saving checkpoints.
            if args.save_checkpoint:
                self.checkpoint_log_dir = os.path.join(
                    args.base_log_dir, "checkpoints", datetime.today().strftime("%Y.%m.%d-%H.%M.%S")
                )
                if not os.path.exists(self.checkpoint_log_dir):
                    os.makedirs(self.checkpoint_log_dir, exist_ok=True)
                print(f"Saving checkpoints to {self.checkpoint_log_dir}")

        self.logger = logging.getLogger(args.exp_name)

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.initial_learning_rate},
            {"params": [], "weight_decay": 5e-4, "lr": self.args.initial_learning_rate},
        ]
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        return optimizer

    def train_one_step(
        self,
        embodiment,
        model,
        optimizer,
        step_id,
        batch,
        batch_size,
        predict_head_yaw,
        device,
        visualizer=None,
    ):
        """Run a single training step."""
        with Timer("step/train/unpack_batch") as _:
            sample = unpack_batch(
                embodiment=embodiment,
                batch=batch,
                batch_size=batch_size,
                image_size=self.args.image_size,
                num_history=self.args.num_history,
                data_type=self.args.data_type,
                feature_type=self.args.feature_type,
                add_external_cam=self.args.add_external_cam,
                rgbd_min_depth_threshold=self.args.rgbd_min_depth_threshold,
                device=device,
            )

        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()
        assert sample["gt_gripper_pred"].shape[-1] == sample["gripper_history"].shape[-1]
        assert (
            sample["gt_gripper_pred"].shape[-1]
            == embodiment.get_number_of_items_in_gripper_prediction()[1]
        )
        if (
            get_rank() == 0
            and (step_id + 1) % self.args.viz_freq == 0
            and model.module.vis is not None
        ):
            model.module.vis.enable()

        with Timer("step/train/compute_losses") as _:
            losses, encoded_inputs, cross_attn_weights = model(
                sample["gt_gripper_pred"],
                sample["gt_head_yaw"],
                sample["rgbs"],
                sample["pcds"],
                sample["pcd_valid_mask"],
                sample["vertex_features"],
                sample["vertices"],
                sample["vertices_valid_mask"],
                sample["instr"],
                sample["gripper_history"],
            )
        (total_loss, pos_loss, rot_loss, gripper_loss, head_yaw_loss) = losses

        # Backward pass
        with Timer("step/train/backprop") as _:
            total_loss.backward()

        # Update -- equvalent to gradient accumulation per batch
        if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
            optimizer.step()

        if get_rank() == 0 and step_id % self.args.viz_freq == 0 and model.module.vis is not None:
            # TODO(remos): Remove or fix tensor visualizer
            # model.module.vis.log_tensor_to_wandb(step_id, "train_")
            model.module.vis.disable()

        # Log
        if get_rank() == 0 and (step_id + 1) % self.args.val_freq == 0:
            wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]}, step=step_id)
            wandb.log({"train-loss/total_loss": total_loss}, step=step_id)
            wandb.log({"train-loss/pos_loss": pos_loss}, step=step_id)
            wandb.log({"train-loss/rot_loss": rot_loss}, step=step_id)
            wandb.log({"train-loss/gripper_loss": gripper_loss}, step=step_id)
            if predict_head_yaw:
                wandb.log({"train-loss/head_yaw_loss": head_yaw_loss}, step=step_id)
            self.logger.info(f"[train] Iteration: {step_id}, Total Loss: {total_loss:.4f}")
            self.logger.info(
                "[train] Breakdown: Pos Loss: %.4f, Rot Loss: %.4f, Openness Loss: %.4f, Head Yaw Loss: %.4f"
                % (pos_loss, rot_loss, gripper_loss, head_yaw_loss if predict_head_yaw else 0.0)
            )

        # visualize
        if visualizer is not None:
            encoded_inputs["cross_attn_weights"] = cross_attn_weights
            visualizer.visualize(
                sample=encoded_inputs,
                data_type=self.args.data_type,
            )
            visualizer.run_until_space_pressed()

    # from trajectory only
    @torch.no_grad()
    def evaluate_nsteps(
        self, embodiment, model, loader, batch_size, step_id, num_batches, predict_head_yaw, split
    ):
        """Run a given number of evaluation steps."""
        if num_batches == -1:
            num_eval_batches = len(loader)
        elif num_batches > 0:
            num_eval_batches = min(num_batches, len(loader))
        else:
            raise ValueError("Number of batches for evaluation shall be -1 or greater than 0.")
        print_dist(
            f"Evaluation on {num_eval_batches} batches of totally {len(loader)} in the dataset ({split} split)."
        )
        device = next(model.parameters()).device
        model.eval()
        values = {
            "mean_distance_m": torch.zeros(1, device=device),
            "distance_m_std": torch.zeros(1, device=device),
            "mean_rot_l1": torch.zeros(1, device=device),
            "mean_rot_error_deg": torch.zeros(1, device=device),
            "mean_openness_l1": torch.zeros(1, device=device),
            "mean_distance_m_x": torch.zeros(1, device=device),
            "distance_m_std_x": torch.zeros(1, device=device),
            "mean_distance_m_y": torch.zeros(1, device=device),
            "distance_m_std_y": torch.zeros(1, device=device),
            "mean_distance_m_z": torch.zeros(1, device=device),
            "distance_m_std_z": torch.zeros(1, device=device),
            "mean_total_loss": torch.zeros(1, device=device),
            "mean_pos_loss": torch.zeros(1, device=device),
            "mean_rot_loss": torch.zeros(1, device=device),
            "mean_gripper_loss": torch.zeros(1, device=device),
            "mean_bias_x": torch.zeros(1, device=device),
            "mean_bias_y": torch.zeros(1, device=device),
            "mean_bias_z": torch.zeros(1, device=device),
        }
        if predict_head_yaw:
            values["mean_head_yaw_error_deg"] = torch.zeros(1, device=device)
            values["mean_head_yaw_loss"] = torch.zeros(1, device=device)

        for i, batch in enumerate(loader):
            if num_batches != -1 and i == num_batches:
                break
            sample = unpack_batch(
                embodiment=embodiment,
                batch=batch,
                batch_size=batch_size,
                image_size=self.args.image_size,
                num_history=self.args.num_history,
                data_type=self.args.data_type,
                feature_type=self.args.feature_type,
                add_external_cam=self.args.add_external_cam,
                rgbd_min_depth_threshold=self.args.rgbd_min_depth_threshold,
                device=device,
            )

            inference_timer = Timer("step/eval/inference")
            pred, head_yaw_pred, losses, _, _ = model(
                sample["gt_gripper_pred"],
                sample["gt_head_yaw"],
                sample["rgbs"],
                sample["pcds"],
                sample["pcd_valid_mask"],
                sample["vertex_features"],
                sample["vertices"],
                sample["vertices_valid_mask"],
                sample["instr"],
                sample["gripper_history"],
                run_inference=True,
            )

            # In keypose mode the num predicted steps is always one
            assert pred.shape == (
                batch_size,
                self.args.prediction_horizon,
                embodiment.get_num_grippers(),
                embodiment.get_number_of_items_in_gripper_prediction()[1],
            )
            if predict_head_yaw:
                assert sample["gt_head_yaw"] is not None
                assert head_yaw_pred.shape == (batch_size, self.args.prediction_horizon, 1)
            inference_timer.stop()

            # Compute mean loss
            (total_loss, pos_loss, rot_loss, gripper_loss, head_yaw_loss) = losses
            values["mean_pos_loss"] += pos_loss / num_eval_batches
            values["mean_rot_loss"] += rot_loss / num_eval_batches
            values["mean_gripper_loss"] += gripper_loss / num_eval_batches
            values["mean_total_loss"] += total_loss / num_eval_batches
            if predict_head_yaw:
                values["mean_head_yaw_loss"] += head_yaw_loss / num_eval_batches

            # Compute mean metrics
            metrics = compute_metrics(
                pred,
                head_yaw_pred,
                sample["gt_gripper_pred"],
                sample["gt_head_yaw"],
                predict_head_yaw=predict_head_yaw,
                rotation_form="quaternion",
            )
            values["mean_distance_m"] += metrics["distance_m"] / num_eval_batches
            values["mean_distance_m_x"] += metrics["distance_m_x"] / num_eval_batches
            values["mean_distance_m_y"] += metrics["distance_m_y"] / num_eval_batches
            values["mean_distance_m_z"] += metrics["distance_m_z"] / num_eval_batches
            values["distance_m_std"] += metrics["distance_m_std"] / num_eval_batches
            values["distance_m_std_x"] += metrics["distance_m_std_x"] / num_eval_batches
            values["distance_m_std_y"] += metrics["distance_m_std_y"] / num_eval_batches
            values["distance_m_std_z"] += metrics["distance_m_std_z"] / num_eval_batches
            values["mean_bias_x"] += metrics["bias"][0] / num_eval_batches
            values["mean_bias_y"] += metrics["bias"][1] / num_eval_batches
            values["mean_bias_z"] += metrics["bias"][2] / num_eval_batches
            values["mean_rot_l1"] += metrics["rot_l1"] / num_eval_batches
            values["mean_rot_error_deg"] += metrics["rot_error_deg"] / num_eval_batches
            values["mean_openness_l1"] += metrics["openness_l1"] / num_eval_batches
            if predict_head_yaw:
                values["mean_head_yaw_error_deg"] += (
                    metrics["head_yaw_error_deg"] / num_eval_batches
                )

            # Viz trajectories on the 1st batch
            if get_rank() == 0 and i == 0:
                fig = generate_visualizations(pred, sample["gt_gripper_pred"])
                wandb.log({f"{split}-viz/viz": fig}, step=step_id)

        # calling all_gather API to get the values on all processes
        values = self.synchronize_between_processes(values)
        # all_gather collects tensors from each process and stacks them into a single tensor
        # mean over all processes to report on distributed training
        values = {k: v.mean(dim=0).item() for k, v in values.items()}

        if get_rank() == 0:
            wandb.log({f"{split}-loss/total_loss": values["mean_total_loss"]}, step=step_id)
            wandb.log({f"{split}-loss/pos_loss": values["mean_pos_loss"]}, step=step_id)
            wandb.log({f"{split}-loss/rot_loss": values["mean_rot_loss"]}, step=step_id)
            wandb.log({f"{split}-loss/gripper_loss": values["mean_gripper_loss"]}, step=step_id)
            if predict_head_yaw:
                wandb.log(
                    {f"{split}-loss/head_yaw_loss": values["mean_head_yaw_loss"]}, step=step_id
                )
            wandb.log(
                {f"{split}-metrics/distance_error_m": values["mean_distance_m"]}, step=step_id
            )
            wandb.log(
                {f"{split}-metrics/distance_error_mx": values["mean_distance_m_x"]}, step=step_id
            )
            wandb.log(
                {f"{split}-metrics/distance_error_my": values["mean_distance_m_y"]}, step=step_id
            )
            wandb.log(
                {f"{split}-metrics/distance_error_mz": values["mean_distance_m_z"]}, step=step_id
            )

            wandb.log({f"{split}-metrics/rotation_error_l1": values["mean_rot_l1"]}, step=step_id)
            wandb.log(
                {f"{split}-metrics/rotation_error_deg": values["mean_rot_error_deg"]}, step=step_id
            )
            wandb.log(
                {f"{split}-metrics/openness_error_l1": values["mean_openness_l1"]}, step=step_id
            )
            if predict_head_yaw:
                wandb.log(
                    {f"{split}-metrics/head_yaw_error_deg": values["mean_head_yaw_error_deg"]},
                    step=step_id,
                )
            wandb.log({f"{split}-metrics/distance_std_m": values["distance_m_std"]}, step=step_id)
            wandb.log(
                {f"{split}-metrics/distance_std_mx": values["distance_m_std_x"]}, step=step_id
            )
            wandb.log(
                {f"{split}-metrics/distance_std_my": values["distance_m_std_y"]}, step=step_id
            )
            wandb.log(
                {f"{split}-metrics/distance_std_mz": values["distance_m_std_z"]}, step=step_id
            )

            wandb.log({f"{split}-metrics/bias_x": values["mean_bias_x"]}, step=step_id)
            wandb.log({f"{split}-metrics/bias_y": values["mean_bias_y"]}, step=step_id)
            wandb.log({f"{split}-metrics/bias_z": values["mean_bias_z"]}, step=step_id)
            self.logger.info(
                f'[{split}] Iteration: {step_id}, Total Loss: {values["mean_total_loss"]:.4f}'
            )
            self.logger.info(
                "[%s] Breakdown: Pos Loss: %.4f, Rot Loss: %.4f, Openness Loss: %.4f, Head Yaw Loss: %.4f"
                % (
                    split,
                    values["mean_pos_loss"],
                    values["mean_rot_loss"],
                    values["mean_gripper_loss"],
                    values["mean_head_yaw_loss"] if predict_head_yaw else 0.0,
                )
            )
            self.logger.info(
                "[%s] Metrics: Distance Error: %.4f, Rot Error: %.4f, Openness Error: %.4f, Head Yaw Error: %.4f, Distance STD: %.4f"
                % (
                    split,
                    values["mean_distance_m"],
                    values["mean_rot_error_deg"],
                    values["mean_openness_l1"],
                    values["mean_head_yaw_error_deg"] if predict_head_yaw else 0.0,
                    values["distance_m_std"],
                )
            )
            self.logger.info(
                "[%s] Metrics: Bias: [%.4f, %.4f, %.4f]"
                % (split, values["mean_bias_x"], values["mean_bias_y"], values["mean_bias_z"])
            )
        return values["mean_total_loss"]

    def synchronize_between_processes(self, a_dict):
        all_dicts = all_gather(a_dict)

        if get_rank() != 0:
            merged = {}
            for key in all_dicts[0].keys():
                device = all_dicts[0][key].device
                merged[key] = torch.cat([p[key].to(device) for p in all_dicts if key in p])
            a_dict = merged
        return a_dict

    def log_timings_to_wandb(self, step_id):
        wandb.log({f"timings/mean_step_time_s": get_mean_time("step")}, step=step_id)
        wandb.log({f"timings/step_time_s": get_last_time("step")}, step=step_id)
        wandb.log({f"timings/batch_loading_time_s": get_last_time("step/load_batch")}, step=step_id)
        wandb.log(
            {f"timings/batch_collate_time_s": get_last_time("step/load_batch/collate_batch")},
            step=step_id,
        )
        wandb.log({f"timings/train_step_time_s": get_last_time("step/train")}, step=step_id)
        wandb.log(
            {f"timings/unpack_batch_time_s": get_last_time("step/train/unpack_batch")}, step=step_id
        )
        wandb.log(
            {f"timings/compute_losses_time_s": get_last_time("step/train/compute_losses")},
            step=step_id,
        )
        wandb.log({f"timings/backprop_time_s": get_last_time("step/train/backprop")}, step=step_id)
        wandb.log(
            {f"timings/train_evaluation_time_s": get_last_time("step/eval/train-val")}, step=step_id
        )
        wandb.log({f"timings/test_evaluation_time_s": get_last_time("step/eval/val")}, step=step_id)
        wandb.log({f"timings/inference_time_s": get_mean_time("step/eval/inference")}, step=step_id)
        wandb.log({f"timings/getitem_time_s": get_mean_time("data_engine/getitem")}, step=step_id)

    def log_data_queue_stats_to_wandb(self, step_id):
        shmem_mb = get_shmem_usage_mb()
        wandb.log({f"data_queue/used_shared_memory_mb": shmem_mb}, step=step_id)

    def run_training(self):
        """Run training/testing pipeline."""
        device = "cuda"

        weighting_type = get_sampling_weighting_type(self.args.sampling_weighting_type.upper())

        if self.args.demos_valset is None:
            print_dist("No validation set specified, using train set as validation set.")
            self.args.demos_valset = self.args.demos_train

        if self.args.relative_action:
            print_dist("Using relative action.")
        else:
            print_dist("Using absolute action.")

        if self.args.use_keyposes:
            print_dist("Using keypose mode.")
        else:
            print_dist("Using trajectory mode")

        visualizer = Visualizer(self.args) if self.args.visualize else None

        print_dist(f"Using {weighting_type}.")
        print_dist(f"Using {self.args.data_type} data type.")

        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        train_loader, train_sampler = None, None
        # Initialize the embodiment
        embodiment_type = get_embodiment_type_from_task(self.args.task)
        if embodiment_type == EmbodimentType.ARM:
            embodiment = ArmEmbodiment(self.args, device)
        elif embodiment_type == EmbodimentType.HUMANOID:
            embodiment = HumanoidEmbodiment(self.args.task, self.args, device)
        else:
            raise ValueError(f"Unsupported embodiment type: {embodiment_type}")
        predict_head_yaw = get_predict_head_yaw_from_task(self.args.task)
        if not self.args.eval_only:
            start = time.perf_counter()
            train_loader, train_sampler = get_data_loader_by_data_type(
                embodiment=embodiment,
                dataset_path=self.args.dataset,
                demos=self.args.demos_train,
                task=self.args.task,
                num_workers=self.args.num_workers,
                batch_size=self.args.batch_size,
                use_keyposes=self.args.use_keyposes,
                data_type=self.args.data_type,
                only_sample_keyposes=self.args.only_sample_keyposes,
                extra_keyposes_around_grasp_events=self.args.extra_keyposes_around_grasp_events,
                keypose_detection_mode=self.args.keypose_detection_mode,
                include_failed_demos=self.args.include_failed_demos,
                sampling_weighting_type=weighting_type,
                gripper_encoding_mode=self.args.gripper_encoding_mode,
                num_history=self.args.num_history,
                prediction_horizon=self.args.prediction_horizon,
                apply_random_transforms=bool(self.args.apply_random_transforms),
                apply_geometry_noise=bool(self.args.apply_geometry_noise),
                pos_noise_stddev_m=self.args.pos_noise_stddev_m,
                rot_noise_stddev_deg=self.args.rot_noise_stddev_deg,
                add_external_cam=self.args.add_external_cam,
                num_vertices_to_sample=self.args.num_vertices_to_sample,
                vertex_sampling_method=self.args.vertex_sampling_method,
                random_translation_range_m=self.args.random_translation_range_m,
                random_rpy_range_deg=self.args.random_rpy_range_deg,
                seed=self.args.seed,
            )

            print_dist("Train loader time", time.perf_counter() - start)

        # Check if the user has requested a different number of workers for the test dataset.
        # NOTE(alexmillane): We've exposed this as it allows to eliminate the pre-load queue
        # for validation, significantly reducing the CPU memory usage.
        num_workers_for_test_dataset = self.args.num_workers_for_test_dataset
        if num_workers_for_test_dataset == None or num_workers_for_test_dataset < 0.0:
            num_workers_for_test_dataset = self.args.num_workers

        validation_loader, validation_sampler = get_data_loader_without_augmentations(
            embodiment=embodiment,
            dataset_path=self.args.dataset,
            demos=self.args.demos_valset,
            task=self.args.task,
            num_workers=num_workers_for_test_dataset,
            batch_size=self.args.batch_size_val,
            use_keyposes=self.args.use_keyposes,
            data_type=self.args.data_type,
            extra_keyposes_around_grasp_events=self.args.extra_keyposes_around_grasp_events,
            keypose_detection_mode=self.args.keypose_detection_mode,
            include_failed_demos=self.args.include_failed_demos,
            gripper_encoding_mode=self.args.gripper_encoding_mode,
            num_history=self.args.num_history,
            prediction_horizon=self.args.prediction_horizon,
            add_external_cam=self.args.add_external_cam,
            num_vertices_to_sample=self.args.num_vertices_to_sample,
            vertex_sampling_method=self.args.vertex_sampling_method,
            sampling_weighting_type=SamplingWeightingType.UNIFORM,
            seed=self.args.seed,
        )

        model = get_model(self.args)
        # Calculate the number of parameters
        if get_rank() == 0:
            total_params = print_num_trainable_params(model)
            wandb.config.update({"Model Parameters": total_params})
        optimizer = self.get_optimizer(model)

        # Move model to devices
        assert torch.cuda.is_available(), "CUDA is not available"
        model = model.cuda()

        model = DDP(
            model,
            device_ids=[get_rank()],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        start_iter, best_loss = 0, None
        if self.args.checkpoint:
            start_iter, best_loss = load_train_checkpoint(self.args, model, optimizer)
        if self.args.eval_only:
            print_dist("Test evaluation only.......")
            # num_batches = -1, eval on entire test set
            self.evaluate_nsteps(
                embodiment,
                model,
                validation_loader,
                self.args.batch_size_val,
                step_id=0,
                num_batches=-1,
                predict_head_yaw=predict_head_yaw,
                split="val-only",
            )
            return

        # Set up learning rate scheduler.
        lr_convergence_iter = int(
            self.args.train_iters * self.args.learning_rate_convergence_percentage
        )
        lr_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.args.learning_rate_end_factor,
            total_iters=lr_convergence_iter,
        )

        # Training loop
        train_loader_iter = None
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        # Note (xyao): In distributed mode, calling the set_epoch() method at the
        # beginning of each epoch before creating the DataLoader iterator
        # is necessary to make shuffling work properly across multiple epochs.
        # Otherwise, the same ordering will be always used.
        model.train()

        device = model.device
        train_epoch_length = len(train_loader)
        validation_epoch_length = len(validation_loader)
        assert train_epoch_length != 0, "Train loader contains less than one batch."
        assert validation_epoch_length != 0, "Validation loader contains less than one batch."
        # Eval batches shall use the same order across multi-epoches
        # shuffle only once among all iterations
        if is_dist_avail_and_initialized():
            validation_sampler.set_epoch(0)

        print_dist(f"Train epoch length: {train_epoch_length} batches")
        print_dist(f"Running a total of {self.args.train_iters // train_epoch_length} episodes.")

        # In tqdm, disable=None means that progress bar is disabled for non-tty sessions (e.g. CI)
        for step_id in trange(start_iter, self.args.train_iters, disable=None):
            epoch_idx = step_id // train_epoch_length
            if step_id % self.args.print_progress_freq == 0:
                print_dist("\n-------------------------------------------")
                print_dist(
                    f"Starting step: {step_id} of total {self.args.train_iters} (epoch {epoch_idx})"
                )
            elif not sys.stdout.isatty():
                print_dist(".", end="")
            step_timer = Timer("step")

            # Since it's iter-based training, instead of epoch-based training
            # Shuffle ops is manually set with the estimated epoch

            # Start of a new epoch or a new training
            if step_id % train_epoch_length == 0 or step_id == start_iter:
                print_dist(f"Starting epoch: {epoch_idx}")
                train_loader_iter = None  # reset loader
                # shuffle ops only for multi-gpu, such that each gpu will see different data per epoch
                if get_world_size() > 1:
                    # # Shuffle data among multi-gpus every 5 epochs to reduce shuffle overhead
                    if epoch_idx % 5 == 0:
                        train_sampler.set_epoch(epoch_idx)
                train_loader_iter = iter(train_loader)

            # Load a batch
            with Timer("step/load_batch") as _:
                batch = next(train_loader_iter)

            # Training
            with Timer("step/train") as _:
                self.train_one_step(
                    embodiment,
                    model,
                    optimizer,
                    step_id,
                    batch,
                    self.args.batch_size,
                    predict_head_yaw,
                    device,
                    visualizer,
                )

            # Update the learning rate
            # TODO(remos): step once per epoch once we change to epoch based training.
            lr_scheduler.step()

            # Evaluation
            if (step_id + 1) % self.args.val_freq == 0:
                if not self.args.skip_train_val:
                    print_dist("Train evaluation.......")
                    model.eval()
                    with Timer("step/eval/train-val") as _:
                        # NOTE(alexmillane): Running this code puts additional load on CPU memory.
                        # iter_loaded and train_loader, though derived from the same object, appear
                        # to maintain separate data queues. Disable train-val if CPU memory usage
                        # is a bottle-neck.
                        self.evaluate_nsteps(
                            embodiment,
                            model,
                            train_loader,
                            self.args.batch_size,
                            step_id,
                            self.args.num_batches_per_train_eval,
                            predict_head_yaw,
                            split="train-val",
                        )
                print_dist("Evaluation.......")
                model.eval()
                with Timer("step/eval/val") as _:
                    new_loss = self.evaluate_nsteps(
                        embodiment,
                        model,
                        validation_loader,
                        self.args.batch_size_val,
                        step_id,
                        self.args.num_batches_per_test_eval,
                        predict_head_yaw,
                        split="val",
                    )
                if get_rank() == 0 and self.args.save_checkpoint:
                    best_loss = save_checkpoint(
                        self.checkpoint_log_dir, model, optimizer, step_id, new_loss, best_loss
                    )
                    # Save the args for reproducibility (e.g. rerunning model with same model arguments).
                    self.args.save(Path(self.checkpoint_log_dir) / TRAINING_ARGUMENT_FILE_NAME)
                torch.cuda.synchronize()

            model.train()
            step_timer.stop()

            # Log timings
            if get_rank() == 0:
                self.log_data_queue_stats_to_wandb(step_id)
                self.log_timings_to_wandb(step_id)

            # Print timer status to console
            if step_id % self.args.print_timers_freq == 0:
                print_dist(timer_status_string())

        return model


def main():
    cli_args = TrainingAppArgs().parse_args()
    # Get arguments from CLI and update the model arguments.
    args = update_model_args_from_checkpoint(cli_args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    trainer = Trainer(args)

    trainer.run_training()


if __name__ == "__main__":
    # Run with a multi-process group context manager.
    with MultiProcessGroup():
        main()

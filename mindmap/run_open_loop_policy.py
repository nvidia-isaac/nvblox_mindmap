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
import random
from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from mindmap.cli.args import OpenLoopAppArgs, update_model_args_from_checkpoint
from mindmap.data_loading.batching import unpack_batch
from mindmap.data_loading.data_types import includes_mesh
from mindmap.data_loading.dataset import SamplingWeightingType
from mindmap.data_loading.dataset_files_by_encoding_method import get_data_loader_by_data_type
from mindmap.diffuser_actor.diffuser_actor import DiffuserActor
from mindmap.embodiments.arm.embodiment import ArmEmbodiment
from mindmap.embodiments.embodiment_base import EmbodimentBase, EmbodimentType
from mindmap.embodiments.humanoid.embodiment import HumanoidEmbodiment
from mindmap.embodiments.task_to_embodiment import get_embodiment_type_from_task
from mindmap.model_utils.checkpoint import get_model, load_inference_checkpoint
from mindmap.model_utils.loss import compute_metrics
from mindmap.model_utils.multi_gpu import MultiProcessGroup
from mindmap.model_utils.task_to_predict_head_yaw import get_predict_head_yaw_from_task
from mindmap.visualization.visualization import compute_pca_basis_from_dataset
from mindmap.visualization.visualizer import Visualizer


def run_inference(
    embodiment: EmbodimentBase,
    model: DiffuserActor,
    sample: dict,
    prediction_horizon: int,
    predict_head_yaw: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prediction, head_yaw_pred, _, encoded_inputs, cross_attn_weights = model(
        gt_gripper_pred=None,
        gt_head_yaw=None,
        rgb_obs=sample["rgbs"],
        pcd_obs=sample["pcds"],
        pcd_valid_mask=sample["pcd_valid_mask"],
        vertex_features=sample["vertex_features"],
        vertices=sample["vertices"],
        vertices_valid_mask=sample["vertices_valid_mask"],
        instruction=sample["instr"],
        gripper_history=sample["gripper_history"],
        run_inference=True,
    )

    # Expect only one predicted pose if we're in keypose mode.
    batch_size = 1
    num_grippers = embodiment.get_num_grippers()
    assert prediction.shape == (batch_size, prediction_horizon, num_grippers, 8)
    if predict_head_yaw:
        assert head_yaw_pred.shape == (batch_size, prediction_horizon, 1)
    return prediction, head_yaw_pred, encoded_inputs, cross_attn_weights


def main(args, model: DiffuserActor = None):
    assert args.dataset, "This script requires a dataset"
    embodiment_type = get_embodiment_type_from_task(args.task)
    if embodiment_type == EmbodimentType.ARM:
        embodiment = ArmEmbodiment()
    elif embodiment_type == EmbodimentType.HUMANOID:
        embodiment = HumanoidEmbodiment(args.task)
    else:
        raise ValueError(f"Embodiment type {args.embodiment_type} not supported")
    predict_head_yaw = get_predict_head_yaw_from_task(args.task)
    data_loader, _ = get_data_loader_by_data_type(
        embodiment=embodiment,
        dataset_path=args.dataset,
        demos=args.demos_open_loop,
        task=args.task,
        num_workers=0,
        batch_size=1,
        use_keyposes=args.use_keyposes,
        data_type=args.data_type,
        only_sample_keyposes=args.only_sample_keyposes,
        extra_keyposes_around_grasp_events=args.extra_keyposes_around_grasp_events,
        keypose_detection_mode=args.keypose_detection_mode,
        include_failed_demos=True,
        sampling_weighting_type=SamplingWeightingType.NONE,
        gripper_encoding_mode=args.gripper_encoding_mode,
        num_history=args.num_history,
        prediction_horizon=args.prediction_horizon,
        apply_random_transforms=bool(args.apply_random_transforms),
        apply_geometry_noise=bool(args.apply_geometry_noise),
        pos_noise_stddev_m=args.pos_noise_stddev_m,
        rot_noise_stddev_deg=args.rot_noise_stddev_deg,
        add_external_cam=args.add_external_cam,
        num_vertices_to_sample=args.num_vertices_to_sample,
        vertex_sampling_method=args.vertex_sampling_method,
        random_translation_range_m=args.random_translation_range_m,
        random_rpy_range_deg=args.random_rpy_range_deg,
        seed=args.seed,
    )

    if includes_mesh(args.data_type):
        print(f"Computing PCA basis from features in dataset")
        pca_basis = compute_pca_basis_from_dataset(
            embodiment,
            data_loader,
            args.image_size,
            args.add_external_cam,
            args.data_type,
            args.feature_type,
            args.rgbd_min_depth_threshold,
        )
    else:
        pca_basis = None

    visualizer = Visualizer(args, pca_params=pca_basis)

    sum_distance_m, sum_rot_l1, sum_openness_l1 = (0, 0, 0)
    for i, batch in enumerate(data_loader):
        print(f"Visualize sequence at index {i}")
        sample = unpack_batch(
            embodiment,
            batch,
            batch_size=1,
            image_size=args.image_size,
            num_history=args.num_history,
            data_type=args.data_type,
            feature_type=args.feature_type,
            add_external_cam=args.add_external_cam,
            rgbd_min_depth_threshold=args.rgbd_min_depth_threshold,
            device="cpu",
        )
        sample["add_external_cam"] = args.add_external_cam

        # If we loaded a model, run inference.
        prediction = None
        if args.checkpoint:
            prediction, head_yaw_pred, encoded_inputs, cross_attn_weights = run_inference(
                embodiment,
                model,
                sample,
                args.prediction_horizon,
                predict_head_yaw,
            )

            # Compute and print the metrics.
            metrics = compute_metrics(
                prediction.cpu(),
                head_yaw_pred.cpu() if predict_head_yaw else None,
                sample["gt_gripper_pred"],
                sample["gt_head_yaw"],
                predict_head_yaw=predict_head_yaw,
                rotation_form="quaternion",
            )
            sum_distance_m += metrics["distance_m"]
            sum_rot_l1 += metrics["rot_l1"]
            sum_openness_l1 += metrics["openness_l1"]
            print(f"Metrics: {metrics}")
            sample.update(encoded_inputs)
            sample["cross_attn_weights"] = cross_attn_weights

        visualizer.visualize(sample, args.data_type, prediction)
        if not args.disable_visualizer_wait_on_key:
            visualizer.run_until_space_pressed()

    print(
        f"Mean metrics: distance_m: {sum_distance_m/i}, "
        f"rot_l1: {sum_rot_l1/i}, openness: {sum_openness_l1/i}"
    )


if __name__ == "__main__":
    # Get arguments from CLI and update the model arguments.
    with MultiProcessGroup():
        cli_args = OpenLoopAppArgs().parse_args()
        args = update_model_args_from_checkpoint(cli_args)
        if args.only_sample_keyposes != cli_args.only_sample_keyposes:
            # We allow overriding the only_sample_keyposes argument from the checkpoint model args on the CLI.
            # In open loop this is nice to avoid having to step through every sample, but only run predictions for the keyposes.
            print(
                f"Setting only_sample_keyposes to {cli_args.only_sample_keyposes} (requested on CLI)."
            )
            args.only_sample_keyposes = cli_args.only_sample_keyposes

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        print("Running open-loop visualization with the following args:")
        print(args)

        device = "cuda"
        assert torch.cuda.is_available(), "CUDA is not available"
        model = None

        if args.checkpoint:
            model = get_model(args).to(device)
            model = DDP(
                model,
                device_ids=[int(os.environ["LOCAL_RANK"])],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            model = load_inference_checkpoint(args.checkpoint, model, device=device)

        with torch.no_grad():
            main(args, model)

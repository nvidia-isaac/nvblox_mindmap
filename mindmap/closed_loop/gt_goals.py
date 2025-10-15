# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import pathlib
from typing import List

from mindmap.cli.args import ClosedLoopAppArgs
from mindmap.data_loading.data_types import DataType
from mindmap.data_loading.dataset import SamplingWeightingType, get_dataloader
from mindmap.data_loading.item_names import IS_KEYPOSE_ITEM_NAME, POLICY_STATE_HISTORY_ITEM_NAME
from mindmap.embodiments.embodiment_base import EmbodimentBase
from mindmap.embodiments.state_base import PolicyStateBase


def get_timestep_from_path(path: str) -> int:
    return int(pathlib.Path(path).name.split(".")[0])


def get_gt_goals(
    args: ClosedLoopAppArgs, demo_name: str, embodiment: EmbodimentBase, device: str
) -> List[PolicyStateBase]:
    assert args.dataset != None
    assert pathlib.Path(args.dataset).exists()

    demo_idx_str = demo_name.split("_")[-1]

    # Get the dataloader
    data_loader, _ = get_dataloader(
        dataset_path=args.dataset,
        embodiment=embodiment,
        demos=demo_idx_str,
        task=args.task,
        item_names=[POLICY_STATE_HISTORY_ITEM_NAME, IS_KEYPOSE_ITEM_NAME],
        transforms={},
        num_workers=0,
        batch_size=1,
        use_keyposes=args.use_keyposes,
        only_sample_keyposes=False,
        extra_keyposes_around_grasp_events=args.extra_keyposes_around_grasp_events,
        keypose_detection_mode=args.keypose_detection_mode,
        include_failed_demos=True,
        sampling_weighting_type=SamplingWeightingType.NONE,
        data_type=DataType.RGBD,  # Shouldn't matter (only robot states loaded)
        gripper_encoding_mode=args.gripper_encoding_mode,
        num_history=1,  # We only need the current pose
        prediction_horizon=0,
        seed=0,
    )

    # Extract the gt states
    states = []
    for i, batch in enumerate(data_loader):
        current_pose = batch[POLICY_STATE_HISTORY_ITEM_NAME][0, -1, :]
        if args.use_keyposes:
            batch_is_keypose = batch[IS_KEYPOSE_ITEM_NAME].cpu().item()
            if batch_is_keypose:
                states.append(embodiment.policy_state_type.from_tensor(current_pose.to(device)))
        else:
            # Do subsampling of the ground truth trajectory, but always include last pose
            is_last_pose = i == len(data_loader) - 1
            if i % args.gt_goals_subsampling_factor == 0 or is_last_pose:
                states.append(embodiment.policy_state_type.from_tensor(current_pose.to(device)))

    assert len(states) > 1
    return states

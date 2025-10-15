# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from enum import Enum
from typing import Tuple

import torch


class DemoOutcome(Enum):
    """Success/failure state of a demo"""

    # The demo was successfully generated. Should be equal to 1 for legacy reasons.
    SUCCESS = 1
    # The demo failed to complete during data generation. Should be equal to 0 for legacy reasons.
    FAILED_DATAGEN = 0
    # The demo failed when replaying ground-truth keyposes.
    FAILED_GT_EVAL = -1


def get_move_up_action() -> torch.Tensor:
    """Generate a sequence of actions to move the end effector upward followed by staying still.

    Returns:
        torch.Tensor: A sequence of actions consisting of upward movement followed by staying still.
    """
    move_up_action = torch.tensor([0, 0, 0.1, 0, 0, 0, 0])
    move_up_sequence = move_up_action.unsqueeze(0).repeat(20, 1).unsqueeze(1)
    stay_still_sequence = torch.zeros(10, 1, 7)
    post_mimicgen_sequence = torch.cat((move_up_sequence, stay_still_sequence), dim=0)
    return post_mimicgen_sequence


def compare_states(state_from_dataset, runtime_state, action_index) -> Tuple[bool, str]:
    """Compare states from dataset and runtime to verify matching behavior.

    Args:
        state_from_dataset: State information loaded from the dataset.
        runtime_state: Current state information from the running environment.
        action_index: Index of the action being compared.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if states match within tolerance, False otherwise.
            - str: Detailed log message describing any mismatches found.
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name].squeeze()
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name][
                    action_index
                ]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(
                        f"State shape of {state_name} for asset {asset_name} don't match"
                    )
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.1:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


def demo_directory_from_episode_name(output_dir: str, episode_name: str) -> str:
    """Generate the output directory path for a specific demo episode.

    Args:
        output_dir (str): Base output directory path.
        episode_name (str): Name of the episode.

    Returns:
        str: Complete path to the demo directory.
    """
    demo_idx = int(episode_name.split("_")[-1])
    formatted_demo_name = f"demo_{demo_idx:05d}"
    return f"{output_dir}/{formatted_demo_name}"

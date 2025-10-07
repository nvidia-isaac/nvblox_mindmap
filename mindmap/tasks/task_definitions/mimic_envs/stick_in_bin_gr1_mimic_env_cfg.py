# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from dataclasses import InitVar

from isaaclab.envs.mimic_env_cfg import (
    MimicEnvCfg,
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from isaaclab.utils import configclass

from mindmap.tasks.task_definitions.drill_in_box.config.gr1.mdp.target_side import TargetSide
from mindmap.tasks.task_definitions.stick_in_bin.config.gr1.stick_in_bin_gr1_env_cfg import (
    StickInBinGR1EnvCfg,
)


@configclass
class StickInBinGR1MimicEnvCfg(StickInBinGR1EnvCfg, MimicEnvCfg):
    randomization_size: TargetSide = TargetSide.UNDEFINED

    def __post_init__(self):
        # Calling post init of parents
        super().__post_init__()

        if self.target_side == TargetSide.LEFT:
            eef_name_on_target_side = "left"
            eef_name_on_opposite_side = "right"
        elif self.target_side == TargetSide.RIGHT:
            eef_name_on_target_side = "right"
            eef_name_on_opposite_side = "left"
        else:
            raise ValueError(f"Invalid target side: {self.target_side}")

        # Override the existing values
        self.datagen_config.name = "demo_src_gr1t2_demo_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.num_demo_to_render = 10
        self.datagen_config.num_fail_demo_to_render = 25
        self.datagen_config.seed = 1

        # EEF on target side (arm needs to place the pick up object in the drum)
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="pick_up_object",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal=f"grasp_{eef_name_on_target_side}",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generation
                # selection_strategy="nearest_neighbor_object",
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.005,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="open_drum",
                # Corresponding key for the binary indicator in "datagen_info" for completion
                subtask_term_signal=None,
                # Time offsets for data generation when splitting a trajectory
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.005,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=3,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs[eef_name_on_target_side] = subtask_configs

        # EEF on opposite side (arm is static)
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="pick_up_object",
                # Corresponding key for the binary indicator in "datagen_info" for completion
                subtask_term_signal=None,
                # Time offsets for data generation when splitting a trajectory
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.005,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs[eef_name_on_opposite_side] = subtask_configs

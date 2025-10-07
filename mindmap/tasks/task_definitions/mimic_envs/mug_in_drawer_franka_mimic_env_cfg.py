# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from mindmap.tasks.task_definitions.mug_in_drawer.config.franka.mug_in_drawer_franka_env_cfg import (
    MugInDrawerFrankaEnvCfg,
)


@configclass
class MugInDrawerFrankaMimicEnvCfg(MugInDrawerFrankaEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for MugInDrawer Franka env.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()
        # # TODO: Figure out how we can move this to the MimicEnvCfg class
        # # The __post_init__() above only calls the init for FrankaCubeStackEnvCfg and not MimicEnvCfg
        # # https://stackoverflow.com/questions/59986413/achieving-multiple-inheritance-using-python-dataclasses

        # Override the existing values
        self.datagen_config.name = "demo_src_franka_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # The following are the subtask configurations for the mug in drawer task.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="target_mug",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grasp_1",
                # Specifies time offsets for data generation when splitting a trajectory into
                # subtask segments. Random offsets are added to the termination boundary.
                subtask_term_offset_range=(10, 20),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.03,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="bottom_of_drawer_with_mugs",
                # End of final subtask does not need to be detected
                subtask_term_signal=None,
                # No time offsets for the final subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.03,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["franka"] = subtask_configs

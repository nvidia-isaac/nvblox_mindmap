# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import gymnasium as gym

from . import drill_in_box_gr1_env_cfg
from .mdp.target_side import TargetSide

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Drill-In-Box-GR1T2-Left-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": drill_in_box_gr1_env_cfg.DrillInBoxGR1EnvCfg(
            target_side=TargetSide.LEFT
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Drill-In-Box-GR1T2-Right-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": drill_in_box_gr1_env_cfg.DrillInBoxGR1EnvCfg(
            target_side=TargetSide.RIGHT
        ),
    },
    disable_env_checker=True,
)

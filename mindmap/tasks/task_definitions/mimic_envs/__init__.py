# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

from mindmap.tasks.task_definitions.drill_in_box.config.gr1.drill_in_box_gr1_env_cfg import (
    TargetSide,
)

from .drill_in_box_gr1_mimic_env import DrillInBoxGR1LeftMimicEnv, DrillInBoxGR1RightMimicEnv
from .drill_in_box_gr1_mimic_env_cfg import DrillInBoxGR1MimicEnvCfg
from .mug_in_drawer_franka_mimic_env import MugInDrawerFrankaMimicEnv
from .mug_in_drawer_franka_mimic_env_cfg import MugInDrawerFrankaMimicEnvCfg
from .stick_in_bin_gr1_mimic_env import StickInBinGR1LeftMimicEnv, StickInBinGR1RightMimicEnv
from .stick_in_bin_gr1_mimic_env_cfg import StickInBinGR1MimicEnvCfg

print("Registering gym environment Isaac-Mug-in-Drawer-Franka-Mimic")
gym.register(
    id="Isaac-Mug-in-Drawer-Franka-Mimic-v0",
    entry_point="mindmap.tasks.task_definitions.mimic_envs:MugInDrawerFrankaMimicEnv",
    kwargs={
        "env_cfg_entry_point": mug_in_drawer_franka_mimic_env_cfg.MugInDrawerFrankaMimicEnvCfg,
    },
    disable_env_checker=True,
)

print("Registering gym environment Isaac-Drill-In-Box-GR1T2-Left-Mimic")
gym.register(
    id="Isaac-Drill-In-Box-GR1T2-Left-Mimic-v0",
    entry_point="mindmap.tasks.task_definitions.mimic_envs:DrillInBoxGR1LeftMimicEnv",
    kwargs={
        "env_cfg_entry_point": drill_in_box_gr1_mimic_env_cfg.DrillInBoxGR1MimicEnvCfg(
            target_side=TargetSide.LEFT
        ),
    },
    disable_env_checker=True,
)

print("Registering gym environment Isaac-Drill-In-Box-GR1T2-Right-Mimic")
gym.register(
    id="Isaac-Drill-In-Box-GR1T2-Right-Mimic-v0",
    entry_point="mindmap.tasks.task_definitions.mimic_envs:DrillInBoxGR1RightMimicEnv",
    kwargs={
        "env_cfg_entry_point": drill_in_box_gr1_mimic_env_cfg.DrillInBoxGR1MimicEnvCfg(
            target_side=TargetSide.RIGHT
        ),
    },
    disable_env_checker=True,
)

print("Registering gym environment Isaac-Stick-In-Bin-GR1T2-Left-Mimic")
gym.register(
    id="Isaac-Stick-In-Bin-GR1T2-Left-Mimic-v0",
    entry_point="mindmap.tasks.task_definitions.mimic_envs:StickInBinGR1LeftMimicEnv",
    kwargs={
        "env_cfg_entry_point": stick_in_bin_gr1_mimic_env_cfg.StickInBinGR1MimicEnvCfg(
            target_side=TargetSide.LEFT
        ),
    },
    disable_env_checker=True,
)

print("Registering gym environment Isaac-Stick-In-Bin-GR1T2-Right-Mimic")
gym.register(
    id="Isaac-Stick-In-Bin-GR1T2-Right-Mimic-v0",
    entry_point="mindmap.tasks.task_definitions.mimic_envs:StickInBinGR1RightMimicEnv",
    kwargs={
        "env_cfg_entry_point": stick_in_bin_gr1_mimic_env_cfg.StickInBinGR1MimicEnvCfg(
            target_side=TargetSide.RIGHT
        ),
    },
    disable_env_checker=True,
)

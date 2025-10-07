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

from . import stack_ik_rel_env_cfg

# For recording demos in IsaacLab we use Isaac-Stack-Cube-Franka-IK-Rel-v0.
# Isaac-Stack-Cube-Franka-With-Cams-IK-Rel-v0 is based on it and adds a wrist and table camera.
gym.register(
    id="Isaac-Stack-Cube-Franka-With-Cams-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_env_cfg.FrankaCubeStackWithCamsEnvCfg,
    },
    disable_env_checker=True,
)

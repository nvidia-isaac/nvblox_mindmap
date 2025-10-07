# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Scene definition
##
@configclass
class StickInBinSceneCfg(InteractiveSceneCfg):
    """Configuration for the stick in bin scene."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING

    stick_in_bin_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/stick_in_bin_scene",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Samples/NvBlox/mindmap/stick_in_bin/stick_in_bin_scene.usd"
        ),
    )
    open_drum = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/open_drum",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[4.08, 2.33, 0.0], rot=[0.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Samples/NvBlox/mindmap/stick_in_bin/assets/drum.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    pick_up_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/pick_up_object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[4.6, 1.75, 0.845], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Samples/NvBlox/mindmap/stick_in_bin/assets/wood_stick.usd",
            activate_contact_sensors=True,
            scale=(0.03, 0.03, 0.03),
        ),
    )


@configclass
class StickInBinEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: StickInBinSceneCfg = StickInBinSceneCfg(
        num_envs=1, env_spacing=10, replicate_physics=True
    )

    terminations = None
    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 5
        self.episode_length_s = 30.0

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

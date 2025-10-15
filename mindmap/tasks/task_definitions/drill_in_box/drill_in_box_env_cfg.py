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
from isaaclab.sensors import CameraCfg, ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Scene definition
##
@configclass
class DrillInBoxSceneCfg(InteractiveSceneCfg):
    """Configuration for the DrillInBox scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    (for franka and gr1) which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING

    # Add the drill in box scene here
    drill_in_box_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/drill_in_box_scene",
        # These positions are hardcoded for the drill in box scene. Its important to keep them.
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[4.420, 1.408, -0.795], rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Samples/NvBlox/mindmap/drill_in_box/drill_in_box_scene.usd"
        ),
    )

    open_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/open_box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.23, -0.5, -0.075], rot=[0.707, 0.0, 0.0, 0.707]
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Samples/NvBlox/mindmap/drill_in_box/assets/open_box.usd",
            scale=(1.25, 1.25, 1.25),
        ),
    )
    closed_box_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/closed_box_1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.15, -0.5, -0.075], rot=[0.707, 0.0, 0.0, 0.707]
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Samples/NvBlox/mindmap/drill_in_box/assets/closed_box.usd",
            scale=(1.25, 1.25, 1.25),
        ),
    )
    closed_box_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/closed_box_2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.23, 0.5, -0.075], rot=[0.707, 0.0, 0.0, 0.707]
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Samples/NvBlox/mindmap/drill_in_box/assets/closed_box.usd",
            scale=(1.25, 1.25, 1.25),
        ),
    )
    closed_box_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/closed_box_3",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.15, 0.5, -0.075], rot=[0.707, 0.0, 0.0, 0.707]
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Samples/NvBlox/mindmap/drill_in_box/assets/closed_box.usd",
            scale=(1.25, 1.25, 1.25),
        ),
    )

    power_drill = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/power_drill",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Samples/NvBlox/mindmap/drill_in_box/assets/power_drill.usd",
            activate_contact_sensors=True,
        ),
    )

    contact_forces_power_drill = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/power_drill", history_length=3, track_air_time=True
    )


@configclass
class DrillInBoxEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: DrillInBoxSceneCfg = DrillInBoxSceneCfg(
        num_envs=1, env_spacing=10, replicate_physics=True
    )

    terminations = None
    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

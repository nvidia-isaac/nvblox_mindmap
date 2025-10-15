# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import copy
from enum import Enum
import tempfile

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg
import isaaclab.controllers.utils as ControllerUtils
from isaaclab.devices.openxr import XrCfg
import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.fourier import GR1T2_CFG
from pink.tasks import FrameTask
import torch

from mindmap.embodiments.humanoid.action import HumanoidEmbodimentAction
from mindmap.embodiments.humanoid.controller import OPEN_HAND_JOINT_STATES
from mindmap.tasks.task_definitions.cube_stacking.mdp import franka_stack_events
from mindmap.tasks.task_definitions.drill_in_box.config.gr1 import mdp
from mindmap.tasks.task_definitions.drill_in_box.config.gr1.mdp.target_side import TargetSide
from mindmap.tasks.task_definitions.drill_in_box.drill_in_box_env_cfg import DrillInBoxEnvCfg
from mindmap.tasks.task_definitions.mug_in_drawer.mdp import mug_in_drawer_events


def permute_boxes_left_event_term(target_side: TargetSide):
    left_poses = [(0.23, 0.5, -0.075, 0.0, 0.0, 1.57), (-0.15, 0.5, -0.075, 0.0, 0.0, 1.57)]
    if target_side == TargetSide.LEFT:
        left_box_cfgs = [SceneEntityCfg("open_box"), SceneEntityCfg("closed_box_1")]
    elif target_side == TargetSide.RIGHT:
        left_box_cfgs = [SceneEntityCfg("closed_box_2"), SceneEntityCfg("closed_box_3")]
    else:
        raise ValueError(f"Invalid target side: {target_side}")
    return EventTerm(
        func=mug_in_drawer_events.permute_object_poses,
        mode="reset",
        params={
            "pose_selection_list": left_poses,
            "asset_cfgs": left_box_cfgs,
        },
    )


def permute_boxes_right_event_term(target_side: TargetSide):
    right_poses = [(0.23, -0.5, -0.075, 0.0, 0.0, 1.57), (-0.15, -0.5, -0.075, 0.0, 0.0, 1.57)]
    if target_side == TargetSide.LEFT:
        right_box_cfgs = [SceneEntityCfg("closed_box_2"), SceneEntityCfg("closed_box_3")]
    elif target_side == TargetSide.RIGHT:
        right_box_cfgs = [SceneEntityCfg("open_box"), SceneEntityCfg("closed_box_1")]
    else:
        raise ValueError(f"Invalid target side: {target_side}")
    return EventTerm(
        func=mug_in_drawer_events.permute_object_poses,
        mode="reset",
        params={
            "pose_selection_list": right_poses,
            "asset_cfgs": right_box_cfgs,
        },
    )


def randomize_power_drill_pose_event_term():
    return EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.55, 0.60),
                "y": (-0.07, 0.07),
                "z": (0.32, 0.32),
                "roll": (-1.57, -1.57),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, -3.14),
            },
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("power_drill")],
        },
    )


@configclass
class EventTargetRightCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_power_drill_pose = randomize_power_drill_pose_event_term()
    permute_boxes_left = permute_boxes_left_event_term(TargetSide.RIGHT)
    permute_boxes_right = permute_boxes_right_event_term(TargetSide.RIGHT)


@configclass
class EventTargetLeftCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_power_drill_pose = randomize_power_drill_pose_event_term()
    permute_boxes_left = permute_boxes_left_event_term(TargetSide.LEFT)
    permute_boxes_right = permute_boxes_right_event_term(TargetSide.LEFT)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pink_ik_cfg = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ],
        # Joints to be locked in URDF
        ik_urdf_fixed_joint_names=[
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_knee_pitch_joint",
            "right_knee_pitch_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
            "waist_yaw_joint",
            "waist_pitch_joint",
            "waist_roll_joint",
            "head_yaw_joint",
            "head_roll_joint",
            "head_pitch_joint",
        ],
        hand_joint_names=[
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
            "head_yaw_joint",
        ],
        # the robot in the sim scene we are controlling
        asset_name="robot",
        # Configuration for the IK controller
        # The frames names are the ones present in the URDF file
        # The urdf has to be generated from the USD that is being used in the scene
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=23,
            show_ik_warnings=False,
            variable_input_tasks=[
                FrameTask(
                    "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                    position_cost=1.0,  # [cost] / [m]
                    orientation_cost=0.5,  # [cost] / [rad]
                    lm_damping=50,  # dampening for solver for step jumps
                    gain=0.05,
                ),
                FrameTask(
                    "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                    position_cost=1.0,  # [cost] / [m]
                    orientation_cost=0.5,  # [cost] / [rad]
                    lm_damping=50,  # dampening for solver for step jumps
                    gain=0.05,
                ),
            ],
            fixed_input_tasks=[],
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(
            func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        robot_root_rot = ObsTerm(
            func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_left_eef_pos)
        left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat)
        right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)

        hand_joint_state = ObsTerm(func=mdp.get_hand_state)
        head_joint_state = ObsTerm(func=mdp.get_head_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfgRight:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.2, "asset_cfg": SceneEntityCfg("power_drill")},
    )

    object_too_close_to_robot = DoneTerm(
        func=mdp.object_too_close_to_robot,
        params={"object_cfg": SceneEntityCfg("power_drill"), "min_dist": 0.2},
    )

    success = DoneTerm(
        func=mdp.object_in_box,
        params={"target_side": TargetSide.RIGHT, "max_object_termination_vel_m_s": 0.1},
    )


@configclass
class TerminationsCfgLeft:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.2, "asset_cfg": SceneEntityCfg("power_drill")},
    )

    object_too_close_to_robot = DoneTerm(
        func=mdp.object_too_close_to_robot,
        params={"object_cfg": SceneEntityCfg("power_drill"), "min_dist": 0.2},
    )

    success = DoneTerm(
        func=mdp.object_in_box,
        params={"target_side": TargetSide.LEFT, "max_object_termination_vel_m_s": 0.1},
    )


def get_gr1_actuator_cfg_with_increased_damping():
    actuator_cfg = copy.deepcopy(GR1T2_CFG.actuators)

    # Keep the intermediate and distal joints cfgs as defined in the original cfg
    actuator_cfg["left-hand"].joint_names_expr = ["L_.*_intermediate_.*", "L_.*_distal_.*"]
    actuator_cfg["right-hand"].joint_names_expr = ["R_.*_intermediate_.*", "R_.*_distal_.*"]

    # Add proximal joints with increased damping
    PROXIMAL_JOINT_DAMPING = 1718.0
    actuator_cfg["left-hand-proximal"] = ImplicitActuatorCfg(
        joint_names_expr=[
            "L_.*_proximal_.*",
        ],
        effort_limit=None,
        velocity_limit=None,
        stiffness=None,  # default is 17184
        damping=PROXIMAL_JOINT_DAMPING,  # 1 / 10 of stiffness
    )
    actuator_cfg["right-hand-proximal"] = ImplicitActuatorCfg(
        joint_names_expr=[
            "R_.*_proximal_.*",
        ],
        effort_limit=None,
        velocity_limit=None,
        stiffness=None,  # default is 17184
        damping=PROXIMAL_JOINT_DAMPING,  # 1 / 10 of stiffness
    )

    # Increase head damping
    actuator_cfg["head"].damping = 220.0

    return actuator_cfg


@configclass
class DrillInBoxGR1EnvCfg(DrillInBoxEnvCfg):
    target_side: TargetSide = TargetSide.UNDEFINED

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # simulation settings taken from:
        # https://isaac-sim.github.io/IsaacLab/v2.1.0/source/how-to/cloudxr_teleoperation.html#optimize-xr-performance
        self.sim.dt = 1 / 120
        self.sim.render_interval = 2

        # Set events and terminations based on target side
        if self.target_side == TargetSide.RIGHT:
            self.events = EventTargetRightCfg()
            self.terminations = TerminationsCfgRight()
        elif self.target_side == TargetSide.LEFT:
            self.events = EventTargetLeftCfg()
            self.terminations = TerminationsCfgLeft()
        else:
            raise ValueError(f"Invalid target side: {self.target_side}")

        # Set actions
        self.actions = ActionsCfg()

        # Set observations
        self.observations = ObservationsCfg()

        # Set GR1 as robot
        L_OPEN_HAND_JOINT_STATES = {f"L_{k}": v for k, v in OPEN_HAND_JOINT_STATES.items()}
        R_OPEN_HAND_JOINT_STATES = {f"R_{k}": v for k, v in OPEN_HAND_JOINT_STATES.items()}
        self.scene.robot = GR1T2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.1, 0.0, 0.13),
                rot=(1.0, 0, 0, 0),
                joint_pos={
                    # right-arm
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": 0.0,
                    "right_shoulder_yaw_joint": 0.0,
                    "right_elbow_pitch_joint": -1.5708,
                    "right_wrist_yaw_joint": 0.0,
                    "right_wrist_roll_joint": 0.0,
                    "right_wrist_pitch_joint": 0.0,
                    # left-arm
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_elbow_pitch_joint": -1.5708,
                    "left_wrist_yaw_joint": 0.0,
                    "left_wrist_roll_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    # --
                    "head_.*": 0.0,
                    "waist_.*": 0.0,
                    ".*_hip_.*": 0.0,
                    ".*_knee_.*": 0.0,
                    ".*_ankle_.*": 0.0,
                    **L_OPEN_HAND_JOINT_STATES,
                    **R_OPEN_HAND_JOINT_STATES,
                },
                joint_vel={".*": 0.0},
            ),
            actuators=get_gr1_actuator_cfg_with_increased_damping(),
        )

        # Add semantics
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add the cams
        self.scene.robot_pov_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_yaw_link/RobotPOVCam",
            update_period=0.0,
            height=512,
            width=512,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.01, 1.0e5)),
            offset=CameraCfg.OffsetCfg(
                pos=(0.12515, 0.0, 0.06776),
                rot=(0.62, 0.32, -0.32, -0.63),
                convention="opengl",
            ),
        )

        # Set external view camera
        self.scene.external_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/external_cam",
            update_period=0.0333,
            height=512,
            width=512,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=[1.42, 0.0, 1.2], rot=[0.627, 0.327, 0.327, 0.627], convention="opengl"
            ),
        )
        # Position of the anchor (viewpoint) within the XR frame.
        # The XR frame is aligned with +y and does not change through the XrCfg but only where the xr view is spawned.
        # This should be set to align with the robot angle in the world frame and depends on the scene setup.
        self.xr: XrCfg = XrCfg(
            anchor_pos=(0.1, 0.0, -0.8),
            anchor_rot=(0.7071, 0.0, 0.0, -0.7071),
        )

        # Temporary directory for URDF files
        self.temp_urdf_dir = tempfile.gettempdir()

        # Idle action to hold robot in default pose
        idle_action = HumanoidEmbodimentAction(
            W_t_W_LeftEef=torch.tensor([-0.22878, 0.2536, 1.0953]),
            q_wxyz_W_LeftEef=torch.tensor([0.5, 0.5, -0.5, 0.5]),
            W_t_W_RightEef=torch.tensor([0.22878, 0.2536, 1.0953]),
            q_wxyz_W_RightEef=torch.tensor([0.5, 0.5, -0.5, 0.5]),
            head_yaw_rad=torch.tensor([0.0]),
            left_hand_joint_states=torch.tensor(list(OPEN_HAND_JOINT_STATES.values())),
            right_hand_joint_states=torch.tensor(list(OPEN_HAND_JOINT_STATES.values())),
        )
        self.idle_action = idle_action.to_tensor(include_head_yaw=True)

        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )
        ControllerUtils.change_revolute_to_fixed(
            temp_urdf_output_path, self.actions.pink_ik_cfg.ik_urdf_fixed_joint_names
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
        self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path

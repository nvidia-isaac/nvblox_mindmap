# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import os
from typing import Optional

import gymnasium as gym
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tap import Tap

from mindmap.embodiments.embodiment_base import EmbodimentType
from mindmap.embodiments.task_to_embodiment import get_embodiment_type_from_task
from mindmap.isaaclab_utils.render_settings import RenderSettings


def _get_env_name(args: Tap) -> str:
    """
    Get the environment name from the command line arguments.

    Args:
        args (ClosedLoopAppArgs): The command line arguments.

    Returns:
        str: The environment name."""
    # Dataset file path
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args.hdf5_file)

    # Loading the environment configuration
    env_name = dataset_file_handler.get_env_name()
    if args.task is not None and env_name != args.task.to_full_task_name():
        print(
            f"Environment name requested {args.task.to_full_task_name()} does not match"
            f"the hdf5 file name {env_name}. Using command line environment name {args.task.to_full_task_name()}."
        )
        env_name = args.task.to_full_task_name()
    return env_name


class SimEnvironment:
    """Context manager for creating and destroying sim environment."""

    def __init__(
        self,
        args: Tap,
        absolute_mode: bool = False,
        record_camera_params: Optional[dict] = None,
    ):
        """
        Args:
            args (Tap): Command line arguments.
            absolute_mode (bool): Whether to use absolute mode for poses. This is typically used for closed loop policies.
            record_camera_params (dict): Parameters for the recording camera. None if no recording is desired.
        """
        self.args = args
        self.record_camera_params = record_camera_params
        self.env = None
        self.absolute_mode = absolute_mode

    def __enter__(self):
        return self.create_env()

    def create_env(self):
        env_name = _get_env_name(self.args)
        env_cfg = parse_env_cfg(
            env_name,
            device=self.args.sim_device,
            num_envs=self.args.num_envs,
            use_fabric=not self.args.disable_fabric,
        )
        embodiment_type = get_embodiment_type_from_task(self.args.task)

        # Add recording camera
        if self.record_camera_params is not None:
            env_cfg.scene.record_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/record_cam",
                update_period=0.0333,
                height=1200,
                width=1200,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=self.record_camera_params["focal_length"],
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=self.record_camera_params["position"],
                    rot=self.record_camera_params["rotation"],
                    convention="opengl",
                ),
            )

            if self.args.record_camera_output_path is not None:
                # Ensure directory exists
                os.makedirs(self.args.record_camera_output_path, exist_ok=True)

        # Disable all recorders and terminations
        env_cfg.recorders = {}
        env_cfg.terminations = {}

        # Change the environment from MimicGen -> Perceptive IL config
        env_cfg = _update_environment_config_for_peceptive_il(
            embodiment_type=embodiment_type,
            env_cfg=env_cfg,
            absolute_mode=self.absolute_mode,
            render_settings=self.args.render_settings,
        )

        # create environment from loaded config
        self.env = gym.make(env_name, cfg=env_cfg)

        # reset environment
        self.env.reset(seed=10)

        return self

    def destroy_env(self):
        self.env.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing environment")
        self.destroy_env()


def _update_environment_config_for_peceptive_il(
    embodiment_type: EmbodimentType,
    env_cfg: ManagerBasedRLEnvCfg,
    absolute_mode: bool = False,
    render_settings: RenderSettings = RenderSettings.DEFAULT,
) -> ManagerBasedRLEnvCfg:
    """Update MimicGen environment config with the changes required for Perceptive IL.

    Args:
        env_cfg (FrankaCubeStackEnvCfg): The base configuration loaded from the MimicGen dataset.
        task (Tasks): The task to update the environment config for.
        relative_mode (bool, optional): Control the arm in relative mode. Defaults to False.
        render_settings (RenderSettings, optional): The render settings to use. Defaults to RenderSettings.DEFAULT.
    Returns:
        FrankaCubeStackEnvCfg: The modified environment config.
    """
    if embodiment_type == EmbodimentType.ARM:
        if absolute_mode:
            env_cfg.actions.arm_action.controller.use_relative_mode = False
            env_cfg.actions.arm_action.scale = 1.0
            # Correct a mistake in the offset of the control frame
            # NOTE(alexmillane): I don't want to correct this mistake in relative mode
            # because I guess the MimicGen data was recorded with the mistake in the
            # control-frame that I correct here. So fixing this in relative mode
            # may break MimixGen playback
            # TODO(alexmillane): Remove this once the bug is removed from MimicGen.
            env_cfg.actions.arm_action.body_offset.pos = [0.0, 0.0, 0.1034]
            # Check that this offset in the controller matches the same offset
            # in the thing measuring the eef frame.
            eef_frame = env_cfg.scene.ee_frame.target_frames[0]
            assert (
                eef_frame.offset.pos == env_cfg.actions.arm_action.body_offset.pos
            ), "eef control and measurement frame should have the same offset."
            # Stiffness 400.0 -> 2000.0
            env_cfg.scene.robot.actuators["panda_shoulder"].stiffness = 2000.0
            env_cfg.scene.robot.actuators["panda_forearm"].stiffness = 2000.0
            # Dampling 80.0 -> 240.0
            env_cfg.scene.robot.actuators["panda_shoulder"].damping = 240.0
            env_cfg.scene.robot.actuators["panda_forearm"].damping = 240.0
    elif embodiment_type == EmbodimentType.HUMANOID:
        pass
    else:
        raise ValueError(f"Invalid embodiment type: {embodiment_type}")

    # Move viewer's camera closer to the robot
    env_cfg.viewer.eye = (1.5, 1.5, 1.5)

    # Apply render settings
    if render_settings == RenderSettings.DETERMINISTIC:
        print("Applying deterministic render settings")
        env_cfg.sim.render.antialiasing_mode = "Off"
    elif render_settings == RenderSettings.DEFAULT:
        pass
    elif render_settings == RenderSettings.HIGH_QUALITY:
        env_cfg.sim.render.carb_settings = {"rtx.rendermode": "PathTracing"}
    else:
        raise ValueError(f"Invalid render settings: {render_settings}")

    return env_cfg

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack.config.franka import stack_ik_rel_env_cfg


@configclass
class FrankaCubeStackWithCamsEnvCfg(stack_ik_rel_env_cfg.FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Add semantics
        self.scene.robot.spawn.semantic_tags = [("class", "robot_arm")]

        # Set wrist camera
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0333,
            height=512,
            width=512,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 1.0e5),
            ),
            # NOTE(remos): We update the camera position to have an occluded view of the scene
            # when a cube is grasped. This is to showcase spatial memory capabilities of our model.
            # offset=CameraCfg.OffsetCfg(pos=[0.13, 0.0, -0.15],
            offset=CameraCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.05], rot=[-0.70614, 0.03701, 0.03701, -0.70614], convention="ros"
            ),
        )

        # Set table view camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
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
                pos=[1.0, 0.0, 0.4], rot=[0.35355, -0.61237, -0.61237, 0.35355], convention="ros"
            ),
        )

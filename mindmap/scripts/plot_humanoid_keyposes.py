# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import argparse
import glob
import pathlib
import sys

import numpy as np
import torch

from mindmap.embodiments.humanoid.keypose_estimation import HumanoidEmbodimentKeyposeEstimator
from mindmap.embodiments.humanoid.robot_state import HumanoidEmbodimentRobotState
from mindmap.keyposes.keypose_detection_mode import KeyposeDetectionMode


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze keyposes from demonstration data")
    parser.add_argument("--demo_path", type=pathlib.Path, help="Path to demonstration directory")
    return parser.parse_args()


def main(data_path: pathlib.Path, plot: bool = True) -> int:
    if "demo" in data_path.name:
        print(f"Analyzing demo {data_path}")
        demo_paths = [data_path]
    else:
        demo_paths = [pathlib.Path(p) for p in sorted(glob.glob(str(data_path / "demo_*")))]
        if len(demo_paths) == 0:
            raise ValueError(f"No demo paths found in {data_path}")
        print(f"Found {len(demo_paths)} demos in {data_path}")

    # For each demo
    for demo_path in demo_paths:
        print(f"Analyzing demo at: {demo_path}")
        # Load robot states
        robot_states = []
        for gripper_state_path in sorted(glob.glob(str(demo_path / "*.robot_state.npy"))):
            gripper_state_np = np.load(gripper_state_path)
            gripper_state_tensor = torch.from_numpy(gripper_state_np).to("cpu")
            robot_state = HumanoidEmbodimentRobotState.from_tensor(gripper_state_tensor)
            robot_states.append(robot_state)

        # Extract and plot the keypose indices
        keypose_estimator = HumanoidEmbodimentKeyposeEstimator()
        keypose_estimator.extract_keypose_indices(
            robot_states,
            extra_keyposes_around_grasp_events=[],
            keypose_detection_mode=KeyposeDetectionMode.NONE,
            plot=plot,
        )

    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args.demo_path))

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import importlib
import subprocess

from mindmap.common_utils.system import get_random_port_in_unassigned_range
from mindmap.tests.utils.constants import TestDataLocations

TIMEOUT_S = 300


def test_arm_gt_policy():
    module_name = "mindmap.run_closed_loop_policy"
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"Cannot find the module: {module_name}")
    script_path = spec.origin
    print(f"Found closed-loop script at: {script_path}")

    from mindmap.tests.test_e2e import FRANKA_DATASET_OUTPUT_DIR

    try:
        result = subprocess.run(
            ["torchrun"]
            + [
                "--nnodes=1",
                "--nproc_per_node=1",
                "--master_port",
                f"{get_random_port_in_unassigned_range()}",
            ]
            + [str(script_path)]
            + [
                "--headless",
                "--num_envs",
                "1",
                "--task",
                "cube_stacking",
                "--demos_closed_loop",
                "0",
                "--hdf5_file",
                f"{TestDataLocations.cube_stacking_hdf5_filepath}",
                "--demo_mode",
                "execute_gt_goals",
                "--use_keyposes",
                "1",
                "--wandb_mode",
                "offline",
                "--dataset",
                f"{FRANKA_DATASET_OUTPUT_DIR}",
            ],
            check=True,
            timeout=TIMEOUT_S,
        )
        assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
    except subprocess.TimeoutExpired as e:
        print(f"Process timed out after {TIMEOUT_S} seconds")
        raise

    print("Test passed.")

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
import os
import subprocess

from mindmap.common_utils.system import get_random_port_in_unassigned_range
from mindmap.tests.utils.constants import TestDataLocations

TIMEOUT_S = 300

# This should be printed by the simulation app when an exception is raised.
EXPECTED_OUTPUT = "Exception caught in SimulationAppContext"


def test_isaaclab_shutdown():
    """Launch isaaclab with invalid dataset to trigger an exception. Check that return code is as expected."""
    module_name = "mindmap.run_closed_loop_policy"
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"Cannot find the module: {module_name}")
    script_path = spec.origin
    print(f"Found closed-loop script at: {script_path}")

    try:
        print("Launching isaaclab with invalid dataset to trigger an exception.")
        result = subprocess.run(
            ["torchrun"]
            + [
                "--nnodes=1",
                "--nproc_per_node=1",
                "--master_port",
                f"{get_random_port_in_unassigned_range  ()}",
            ]
            + [str(script_path)]
            + [
                "--headless",
                "--num_envs=1",
                "--hdf5_file=INVALID_PATH",
                "--wandb_mode=offline",
                "--dataset=INVALID_PATH",
                "--demo_mode=closed_loop_wait",
            ],
            check=False,
            timeout=TIMEOUT_S,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "Expected non-zero return code"
        assert (
            EXPECTED_OUTPUT in result.stdout
        ), f"Expected {EXPECTED_OUTPUT} in stdout, but got {result.stdout}"
    except subprocess.TimeoutExpired as e:
        print(f"Process timed out after {TIMEOUT_S} seconds")
        raise

    print("Test passed.")

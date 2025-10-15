# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from datetime import datetime
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

from mindmap.cli.args import (
    CLOSED_LOOP_ARGUMENT_FILE_NAME,
    ClosedLoopAppArgs,
    update_model_args_from_checkpoint,
)
from mindmap.closed_loop.closed_loop_mode import ClosedLoopMode
from mindmap.common_utils.demo_selection import get_episode_names
from mindmap.embodiments.task_to_embodiment import get_embodiment_type_from_task
from mindmap.isaaclab_utils.isaaclab_datagen_utils import (
    DemoOutcome,
    demo_directory_from_episode_name,
)
from mindmap.isaaclab_utils.isaaclab_writer import IsaacLabWriter
from mindmap.isaaclab_utils.simulation_app import SimulationAppContext
from mindmap.model_utils.distributed_training import get_rank
from mindmap.model_utils.multi_gpu import MultiProcessGroup
from mindmap.tasks.tasks import Tasks

"""Run closed loop with GT keyposes on a dataset and overwrite success outcome.

Some demos fail when replaying extracted keyposes, due to minor discrepancies between full
trajectory and keypose mode.  This script allows us to exclude such demos from training.
"""


def main():
    assert get_rank() == 0, "Closed-loop expects to be run in single GPU mode."
    args = ClosedLoopAppArgs().parse_args()

    # Launch the simulator. The context manager ensures that the app is closed also if an error occurs.
    with SimulationAppContext(headless=args.headless, enable_cameras=True) as simulation_app:
        # Run the closed loop policy.
        from mindmap.closed_loop.closed_loop_policy import run_closed_loop_policy
        from mindmap.isaaclab_utils.environments import SimEnvironment

        with SimEnvironment(args, absolute_mode=True) as sim_environment:
            # Initialize the policy
            print("Initializing policy...")
            from mindmap.closed_loop.policies.ground_truth_policy import GroundTruthPolicy

            # We're running ground truth policy to validate the demos
            policy = GroundTruthPolicy(args, device="cuda")

            # Run the policy.
            assert args.num_retries == 1, "Only one retry is supported for demo validation."
            eval_dict = run_closed_loop_policy(
                policy=policy,
                env=sim_environment.env,
                simulation_app=simulation_app,
                args=args,
                log_to_wandb_after_each_demo=False,
                log_to_wandb_after_all_demos=False,
            )

            num_passed = 0
            num_failed = 0
            selected_demos = get_episode_names(args.demos_closed_loop)
            for demo in selected_demos:
                key = f"{demo}_{0}"
                demo_failed = not eval_dict[key]["success"]
                if demo_failed:
                    print(f"Demo {demo} failed validation.")
                    demo_directory = demo_directory_from_episode_name(args.dataset, demo)
                    IsaacLabWriter(demo_directory).write_outcome(DemoOutcome.FAILED_GT_EVAL)
                    num_failed += 1
                else:
                    print(f"Demo {demo} passed validation.")
                    num_passed += 1

            print("Finished demo validation.")
            print(f"Number of passed demos: {num_passed}")
            print(f"Number of failed demos: {num_failed}")
            print(f"Success rate: {float(num_passed) / float(num_passed + num_failed)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

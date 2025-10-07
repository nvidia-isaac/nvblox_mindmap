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

from isaaclab.app import AppLauncher
import torch
import wandb

from mindmap.cli.args import (
    CLOSED_LOOP_ARGUMENT_FILE_NAME,
    ClosedLoopAppArgs,
    update_model_args_from_checkpoint,
)
from mindmap.embodiments.task_to_embodiment import get_embodiment_type_from_task
from mindmap.isaaclab_utils.simulation_app import SimulationAppContext
from mindmap.model_utils.distributed_training import get_rank
from mindmap.model_utils.multi_gpu import MultiProcessGroup
from mindmap.tasks.tasks import Tasks

TASK_TO_RECORDING_CAMERA = {
    Tasks.CUBE_STACKING.name: {
        "focal_length": 18.0,
        "position": (1.0, -0.25, 0.9),
        "rotation": (0.755, 0.354, 0.228, 0.502),
    },
    Tasks.MUG_IN_DRAWER.name: {
        "focal_length": 13.0,
        "position": (-0.76, 0.58, 0.89),
        "rotation": (0.51, 0.23, -0.41, -0.73),
    },
    Tasks.DRILL_IN_BOX.name: {
        "focal_length": 13.0,
        "position": (-0.5, 0.0, 1.75),
        "rotation": (0.683, 0.183, -0.183, -0.683),
    },
    Tasks.STICK_IN_BIN.name: {
        "focal_length": 13.0,
        "position": (4.6, 1.2, 1.9),
        "rotation": (0.92, 0.38, 0.0, 0.0),
    },
}


def main():
    # Get arguments from CLI and update the model arguments.
    cli_args = ClosedLoopAppArgs().parse_args()
    args = update_model_args_from_checkpoint(cli_args)

    # Setup wandb.
    if not os.path.exists(args.base_log_dir):
        os.makedirs(args.base_log_dir, exist_ok=True)
    wandb_id = None
    if args.wandb_name is not None:
        wandb_id = args.wandb_name + datetime.today().strftime("_%Y.%m.%d-%H.%M.%S")
    wandb.init(
        entity=args.wandb_entity,
        project="mindmap Evaluation",
        dir=args.base_log_dir,
        config=args,
        id=wandb_id,
        mode=args.wandb_mode,
    )

    assert get_rank() == 0, "Closed-loop expects to be run in single GPU mode."

    # CUDA baby
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA is not available"

    # Launch the simulator. The context manager ensures that the app is closed also if an error occurs.
    with SimulationAppContext(headless=args.headless, enable_cameras=True) as simulation_app:
        # Run the closed loop policy.
        # NOTE(remos): Simulation app needs to be launched before importing the closed loop policy.
        # Add external perceptive il tasks
        from mindmap.closed_loop.closed_loop_policy import run_closed_loop_policy
        from mindmap.isaaclab_utils.environments import SimEnvironment
        import mindmap.tasks.task_definitions

        record_camera_params = TASK_TO_RECORDING_CAMERA[args.task.name]
        with SimEnvironment(
            args,
            absolute_mode=True,
            record_camera_params=record_camera_params,
        ) as sim_environment:
            # Initialize the policy
            print("Initializing policy...")
            from mindmap.closed_loop.closed_loop_mode import ClosedLoopMode
            from mindmap.closed_loop.policies.goal_policy import get_dummy_policy_for_embodiment
            from mindmap.closed_loop.policies.ground_truth_policy import GroundTruthPolicy
            from mindmap.closed_loop.policies.nvblox_diffuser_actor_policy import (
                NvbloxDiffuserActorPolicy,
            )

            if args.demo_mode == ClosedLoopMode.CLOSED_LOOP_WAIT:
                policy = NvbloxDiffuserActorPolicy(args, device)
            elif args.demo_mode == ClosedLoopMode.EXECUTE_GT_GOALS:
                policy = GroundTruthPolicy(args, device)
            elif args.demo_mode == ClosedLoopMode.DUMMY:
                embodiment_type = get_embodiment_type_from_task(args.task)
                policy = get_dummy_policy_for_embodiment(embodiment_type, device)
            else:
                raise ValueError(f"Invalid mode: {args.demo_mode}")

            # Run the policy.
            run_closed_loop_policy(
                policy=policy,
                env=sim_environment.env,
                simulation_app=simulation_app,
                args=args,
                log_to_wandb_after_each_demo=True,
                log_to_wandb_after_all_demos=False,
                device=device,
            )

            # Save the args for reproducibility.
            if args.eval_file_path is not None:
                argument_file_dir = Path(args.eval_file_path).parent
                args.save(Path(argument_file_dir) / CLOSED_LOOP_ARGUMENT_FILE_NAME)

            # NOTE: This string is used in tests/test_closed_loop.py to determine that the
            # simulation completes successfully. See that file for a discussion.
            print("Finished closed loop execution.")


if __name__ == "__main__":
    # Run with a multi-process group context manager.
    with MultiProcessGroup():
        main()

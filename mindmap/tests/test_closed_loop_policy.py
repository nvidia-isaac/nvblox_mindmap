# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import List, Optional

import torch

from mindmap.cli.args import ClosedLoopAppArgs
from mindmap.embodiments.embodiment_base import EmbodimentType
from mindmap.tests.utils.constants import TestDataLocations
from mindmap.tests.utils.simulation_runner import run_simulation_app_in_separate_process

# Test hdf5 file
TASK_TO_HDF5_FILEPATH = {
    "cube_stacking": TestDataLocations.cube_stacking_hdf5_filepath,
    "drill_in_box": TestDataLocations.drill_in_box_hdf5_filepath,
}

# Test constants
MAX_NUM_STEPS = 300


def _run_closed_loop_dummy_policy(simulation_app, task_name: str, embodiment_type: EmbodimentType):
    # Get arguments from CLI and update the model arguments.
    # To visualize this test, turn off headless mode in:
    # mindmap/tests/utils/simulation_runner.py
    args = ClosedLoopAppArgs().parse_args(
        [
            "--num_envs",
            "1",
            "--task",
            task_name,
            "--demos_closed_loop",
            "0",
            "--hdf5_file",
            f"{TASK_TO_HDF5_FILEPATH[task_name]}",
            "--demo_mode",
            "closed_loop_wait",
            "--use_keyposes",
            "1",
            "--wandb_mode",
            "offline",
            "--terminate_after_n_steps",
            f"{MAX_NUM_STEPS}",
            "--max_num_steps_to_goal",
            f"{MAX_NUM_STEPS + 1}",
            "--visualize_robot_state",
        ]
    )

    # CUDA baby
    device = "cuda"

    # Add external perceptive il tasks
    # Run the closed loop policy.
    # NOTE(remos): Simulation app needs to be launched before importing the closed loop policy.
    from mindmap.closed_loop.closed_loop_policy import run_closed_loop_policy
    from mindmap.isaaclab_utils.environments import SimEnvironment
    import mindmap.tasks.task_definitions  # noqa: F401

    with SimEnvironment(args, absolute_mode=True) as sim_environment:
        # Initialize the policy
        print("Initializing policy...")
        from mindmap.closed_loop.policies.goal_policy import GoalPolicy
        from mindmap.embodiments.arm.policy_state import ArmEmbodimentPolicyState
        from mindmap.embodiments.embodiment_base import EmbodimentBase
        from mindmap.embodiments.humanoid.policy_state import HumanoidEmbodimentPolicyState
        from mindmap.embodiments.observation_base import ObservationBase
        from mindmap.embodiments.state_base import PolicyStateBase

        class TestGoalPolicy(GoalPolicy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def reset(self):
                super().reset()
                self.first_goal = True
                self.goal_reached = False

            def get_new_goal(
                self,
                embodiment: EmbodimentBase,
                current_state: PolicyStateBase,
                observation: ObservationBase,
            ) -> List[Optional[PolicyStateBase]]:
                print(f"Goal requested.")
                if self.first_goal:
                    goal = super().get_new_goal(embodiment, current_state, observation)
                    self.first_goal = False
                    return goal
                else:
                    self.goal_reached = True
                    return [None]

        # Goals per embodiment
        print(f"Getting a goal for {embodiment_type}...")
        if embodiment_type == EmbodimentType.ARM:
            goal = ArmEmbodimentPolicyState(
                W_t_W_Eef=torch.tensor([0.6, 0.25, 0.25], device=device),
                q_wxyz_W_Eef=torch.tensor([0, 1, 0, 0], device=device),
                gripper_closedness=torch.zeros((1), device=device),
            )
        elif embodiment_type == EmbodimentType.HUMANOID:
            goal = HumanoidEmbodimentPolicyState(
                W_t_W_LeftEef=torch.tensor([0.590, 0.296, 0.452], device=device),
                q_wxyz_W_LeftEef=torch.tensor([0.707, 0.0, -0.707, 0.0], device=device),
                left_hand_closedness=torch.zeros((1), device=device),
                W_t_W_RightEef=torch.tensor([0.532, -0.116, 0.554], device=device),
                q_wxyz_W_RightEef=torch.tensor([0.493, 0.0, -0.87, 0.0], device=device),
                right_hand_closedness=torch.zeros((1), device=device),
                head_yaw_rad=torch.zeros((1), device=device),
            )
        else:
            raise ValueError(f"Embodiment type {embodiment_type} not supported.")

        # Run the test policy.
        policy = None
        test_passed = True
        policy = TestGoalPolicy(args=args, device=device, goal_states=[goal], repeat=False)
        run_closed_loop_policy(
            policy=policy,
            env=sim_environment.env,
            simulation_app=simulation_app,
            args=args,
            log_to_wandb_after_each_demo=False,
            log_to_wandb_after_all_demos=False,
            device=device,
        )
        print("Running closed loop policy done.")
        if policy is None:
            print("Policy failed to initialize.")
            test_passed = False
        if not policy.goal_reached:
            print("Policy failed to reach goal.")
            test_passed = False

    return test_passed


def test_arm_closed_loop_dummy_policy():
    res = run_simulation_app_in_separate_process(
        _run_closed_loop_dummy_policy, "cube_stacking", EmbodimentType.ARM
    )
    assert res, "ARM policy failed to reach the goal."


def test_humanoid_closed_loop_dummy_policy():
    res = run_simulation_app_in_separate_process(
        _run_closed_loop_dummy_policy, "drill_in_box", EmbodimentType.HUMANOID
    )
    assert res, "HUMANOID policy failed to reach the goal."

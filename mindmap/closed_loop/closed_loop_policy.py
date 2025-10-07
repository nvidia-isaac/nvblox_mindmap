# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import contextlib
import os
from typing import Dict, List, Optional

import gymnasium as gym
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.datasets import HDF5DatasetFileHandler
import torch
import tqdm

from mindmap.cli.args import ClosedLoopAppArgs
from mindmap.closed_loop.closed_loop_mode import ClosedLoopMode
from mindmap.closed_loop.evaluators.cube_stacking_evaluator import CubeStackingEvaluator
from mindmap.closed_loop.evaluators.evaluator_base import BasicEvaluator, EvaluatorBase
from mindmap.closed_loop.evaluators.mug_in_drawer_evaluator import MugInDrawerEvaluator
from mindmap.closed_loop.policies.ground_truth_policy import GroundTruthPolicy
from mindmap.closed_loop.policies.policy_base import PolicyBase
from mindmap.common_utils.demo_selection import get_episode_names
from mindmap.embodiments.arm.embodiment import ArmEmbodiment
from mindmap.embodiments.embodiment_base import EmbodimentBase, EmbodimentType
from mindmap.embodiments.humanoid.embodiment import HumanoidEmbodiment
from mindmap.embodiments.state_base import PolicyStateBase
from mindmap.embodiments.task_to_embodiment import get_embodiment_type_from_task
from mindmap.isaaclab_utils.simulation_app import SimulationAppContext
from mindmap.tasks.tasks import Tasks
from mindmap.visualization.visualization import VideoWriter

CLOSED_LOOP_DEMO_LENGTH_TRAJECTORY_MODE = 500
CLOSED_LOOP_DEMO_LENGTH_KEYFRAME_MODE = 500
# Run until all gt predictions are consumed in GT predictions mode.
CLOSED_LOOP_DEMO_LENGTH_GT_GOALS_MODE = 10000000

TASK_TO_EVALUATOR_CLASS_DICT = {
    Tasks.CUBE_STACKING.name: CubeStackingEvaluator,
    Tasks.MUG_IN_DRAWER.name: MugInDrawerEvaluator,
    Tasks.DRILL_IN_BOX.name: BasicEvaluator,
    Tasks.STICK_IN_BIN.name: BasicEvaluator,
}


def get_episode_length(args: ClosedLoopAppArgs) -> int:
    """
    Get the episode length based on the arguments.
    """
    if args.terminate_after_n_steps:
        episode_length = args.terminate_after_n_steps
    elif args.demo_mode == ClosedLoopMode.EXECUTE_GT_GOALS:
        episode_length = CLOSED_LOOP_DEMO_LENGTH_GT_GOALS_MODE
    else:
        if args.use_keyposes:
            episode_length = CLOSED_LOOP_DEMO_LENGTH_KEYFRAME_MODE
        else:
            episode_length = CLOSED_LOOP_DEMO_LENGTH_TRAJECTORY_MODE
    return episode_length


def get_embodiment(args: ClosedLoopAppArgs) -> EmbodimentBase:
    """Get the embodiment based on the arguments."""
    embodiment_type = get_embodiment_type_from_task(args.task)
    if embodiment_type == EmbodimentType.ARM:
        embodiment = ArmEmbodiment(args=args)
    elif embodiment_type == EmbodimentType.HUMANOID:
        embodiment = HumanoidEmbodiment(args.task, args)
    else:
        raise ValueError(f"Invalid embodiment type: {embodiment_type}")
    return embodiment


def run_one_episode(
    policy: PolicyBase,
    args: ClosedLoopAppArgs,
    evaluator: EvaluatorBase,
    dataset_file_handler: HDF5DatasetFileHandler,
    current_demo_name: str,
    env: gym.Env,
    retry_idx: int,
    device: str,
):
    """Runs a policy on a single episode in closed loop.

    Args:
        policy (PolicyBase): The policy used for the closed-loop policy.
        args (ClosedLoopAppArgs): The arguments for the closed-loop policy.
        evaluator (EvaluatorBase): The evaluator used for the closed-loop policy.
        dataset_file_handler (HDF5DatasetFileHandler): The dataset file handler used for the closed-loop policy.
        current_demo_name (str): The name of the current demo.
        env (gym.Env): The environment in which the closed-loop policy is being executed.
        mode (ClosedLoopMode): The mode of the closed-loop policy.
        retry_idx (int): The index of the retry.
        device (str): The device on which the closed-loop policy is being executed.
    """
    # Embodiment
    embodiment = get_embodiment(args)

    # Reset the policy's state.
    policy.reset()

    # Read the initial state of the world for this episode.
    episode_data = dataset_file_handler.load_episode(current_demo_name, env.unwrapped.device).data
    initial_state = episode_data["initial_state"]
    env.unwrapped.reset_to(initial_state, None, is_relative=True)

    # Get the initial state
    robot_state = embodiment.get_robot_state(env)
    policy_state = embodiment.get_policy_state_from_embodiment_state(
        robot_state, last_goal_state=None
    )

    # Run inference on the first run, to get things started.
    goal_state: Optional[PolicyStateBase] = None
    steps_to_reach_goal = 0

    # NOTE(alexmillane): If you read image data, without first stepping the
    # env, we get a the first frame as black and white. This caused the system/network
    # to get into a bad state from which it never recovered. The black and white
    # image is bad, but it's also a bad sign that the network is not robust to a
    # single bad frame.
    # TODO(alexmillane): Update this comment when we make the network robust to single
    # bad frames.
    NUM_FRAMES_TO_SKIP = 2
    for i in range(NUM_FRAMES_TO_SKIP):
        # Step the simulator by commanding the end-effector to the current position.
        action = embodiment.get_action_from_policy_state(policy_state)
        env.step(embodiment.convert_action_to_tensor(action).unsqueeze(0))

    # Initialize the evaluator for the current demo (e.g. initial cube positions).
    evaluator.init_demo(current_demo_name, retry_idx, env)

    video_writer = None
    if args.record_videos:
        record_camera = env.unwrapped.scene["record_cam"]
        video_writer = VideoWriter(
            os.path.join(args.record_camera_output_path, f"{current_demo_name}.mp4"),
            fps=15,
            video_size=args.video_size,
        )

    # How many steps to simulate
    episode_length = get_episode_length(args)

    # The groundtruth policy needs demo-specific initialization.
    if isinstance(policy, GroundTruthPolicy):
        policy.init_for_demo(demo_name=current_demo_name, embodiment=embodiment)

    # ----------------------------
    # Execute the requested demos
    # ----------------------------
    goal_reached_flag = False
    new_goal_requested = False
    goal_state_list = []
    is_intermediate_goal = False
    # In tqdm, disable=None means that progress bar is disabled for non-tty sessions (e.g. CI)
    for action_idx in tqdm.tqdm(range(episode_length), disable=None):
        steps_to_reach_goal += 1
        # Get the state of the arm.
        robot_state = embodiment.get_robot_state(env)
        policy_state = embodiment.get_policy_state_from_embodiment_state(
            robot_state, last_goal_state=goal_state
        )

        if args.visualize_robot_state:
            embodiment.visualize_robot_state(robot_state, goal_state=goal_state)

        # Observations
        observation = embodiment.get_observation(env)

        # Check if we've reached our current goal, update the "goal_reached" flag.
        if goal_state is not None:
            goal_reached_flag = embodiment.is_goal_reached(
                policy_state, goal_state, is_intermediate_goal
            )
            if goal_reached_flag:
                print(f"GOAL REACHED. Simulation steps to reach: {steps_to_reach_goal}")

        # If we've reached the max number of steps to goal, request a new goal.
        goal_timeout_flag = steps_to_reach_goal > args.max_num_steps_to_goal
        if goal_timeout_flag:
            print(f"MAX NUM STEPS REACHED: {args.max_num_steps_to_goal}. Requesting new goal.")
            # Print out remaining errors to goal.
            embodiment.is_goal_reached(policy_state, goal_state, print_errors=True)

        # Request inference if goal reached.
        if goal_reached_flag or goal_timeout_flag:
            steps_to_reach_goal = 0
            new_goal_requested = True

        # Update policy
        policy.step(current_state=policy_state, observation=observation)

        # Get new goal from policy
        if new_goal_requested or goal_state is None:
            if len(goal_state_list) == 0:
                print(f"Getting new goal from policy.")
                goal_state_policy_list = policy.get_new_goal(
                    embodiment=embodiment, current_state=policy_state, observation=observation
                )

                # Add intermediate goals if needed.
                goal_state_list, is_intermediate_goal_list = embodiment.add_intermediate_goals(
                    policy_state, goal_state_policy_list
                )
            # Get the next goal from the list.
            goal_state = goal_state_list.pop(0)
            is_intermediate_goal = is_intermediate_goal_list.pop(0)

            if goal_state is None:
                # If the policy returns None, we're done.
                print(f"Policy indicated that we are done. Exiting demo.")
                break
            # Waiting until next request
            new_goal_requested = False

        # Update the environment, by running the action
        assert goal_state is not None
        action = embodiment.get_action_from_policy_state(goal_state)
        env.step(embodiment.convert_action_to_tensor(action).unsqueeze(0))

        # Record from the record camera
        if video_writer is not None:
            video_writer.add_image(record_camera.data.output["rgb"])

        # Evaluate the current simulation step of the demo.
        evaluator.evaluate_step(robot_state, env)

    if video_writer is not None:
        video_writer.close()

    # Finalize the evaluation of the current demo.
    evaluator.finalize_demo(robot_state, env)


def run_closed_loop_policy(
    policy: PolicyBase,
    env: gym.Env,
    simulation_app: SimulationAppContext,
    args: ClosedLoopAppArgs,
    log_to_wandb_after_each_demo: bool,
    log_to_wandb_after_all_demos: bool,
    wandb_step_id: int = None,
    device: str = "cuda",
) -> Dict:
    """
    Runs a closed-loop policy in the Isaac Lab environment.

    Args:
        env (gym.Env): The environment in which the closed-loop policy is being executed.
        simulation_app (SimulationAppContext): The Isaac Lab simulation application.
        args (ClosedLoopAppArgs): The arguments for the closed-loop policy.
        log_to_wandb_after_each_demo (bool): Whether to log to WandB after each demo.
        log_to_wandb_after_all_demos (bool): Whether to log to WandB after all demos have been run.
        wandb_step_id (int, optional): The step ID for WandB logging. Defaults to None.
        device (str, optional): The device to use for the simulation. Defaults to 'cuda'.

    Returns:
        Dict: A dictionary containing the evaluation results.
    """
    # Dataset file path
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args.hdf5_file)

    # Parse the requested demos from a string to a list of names.
    selected_demo_names = get_episode_names(args.demos_closed_loop)
    print("The following demos were selected from CLI:", selected_demo_names)

    # Initialize the evaluator.
    if args.checkpoint is not None:
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[
            0
        ]  # Remove path and extension
    else:
        checkpoint_name = "unknown"
    evaluator = TASK_TO_EVALUATOR_CLASS_DICT[args.task.name](
        args.task, args.eval_file_path, wandb_step_id, checkpoint_name=checkpoint_name
    )

    # ----------------------------
    # Execute the requested demos
    # ----------------------------
    print(f"Running in {args.demo_mode} mode.")
    eval_dict = {}
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            for current_demo_name in dataset_file_handler.get_episode_names():
                if len(selected_demo_names) > 0 and current_demo_name not in selected_demo_names:
                    continue

                # Run a number of retries for one demo.
                print(f"Starting {current_demo_name}.")
                for retry_idx in range(args.num_retries):
                    print(f"Run {retry_idx+1} of {args.num_retries}.")
                    run_one_episode(
                        args=args,
                        policy=policy,
                        evaluator=evaluator,
                        dataset_file_handler=dataset_file_handler,
                        current_demo_name=current_demo_name,
                        env=env,
                        retry_idx=retry_idx,
                        device=device,
                    )

                    # Update the summary after each demo for intermediate results.
                    eval_dict = evaluator.summarize_demos()

                    # Log to wandb (e.g. success rate).
                    if log_to_wandb_after_each_demo:
                        evaluator.log_to_wandb()
            break

    # Log final summary to wandb (e.g. success rate).
    if log_to_wandb_after_all_demos:
        evaluator.log_to_wandb()

    return eval_dict

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
from pathlib import Path
from typing import List

import gymnasium as gym
from isaaclab.app import AppLauncher
import torch
import tqdm

from mindmap.isaaclab_utils.isaaclab_datagen_utils import DemoOutcome

"""The IsaacLab application has to be launched for the imports to be correctly handled."""
from mindmap.cli.args import DATAGEN_ARGUMENT_FILE_NAME, DataGenAppArgs
from mindmap.isaaclab_utils.simulation_app import SimulationAppContext

args = DataGenAppArgs().parse_args()

from isaaclab.utils.datasets import HDF5DatasetFileHandler

from mindmap.common_utils.demo_selection import get_episode_names
from mindmap.data_loading.data_types import (
    DataType,
    includes_depth_camera,
    includes_nvblox,
    includes_rgb,
)
from mindmap.embodiments.arm.embodiment import ArmEmbodiment
from mindmap.embodiments.embodiment_base import EmbodimentBase, EmbodimentType
from mindmap.embodiments.humanoid.embodiment import HumanoidEmbodiment
from mindmap.embodiments.task_to_embodiment import get_embodiment_type_from_task
from mindmap.isaaclab_utils.isaaclab_datagen_utils import (
    compare_states,
    demo_directory_from_episode_name,
    get_move_up_action,
)
from mindmap.isaaclab_utils.isaaclab_writer import IsaacLabWriter
from mindmap.mapping.isaaclab_nvblox_mapper import IsaacLabNvbloxMapper
from mindmap.tasks.tasks import Tasks
from mindmap.visualization.visualizer import Visualizer


class IsaacLabDataGenerator:
    """A class for replaying and processing IsaacLab demonstrations.

    This class handles loading and replaying demonstrations from a dataset,
    processing camera data, and writing the results to output files.

    Args:
        args (DataGenAppArgs): Configuration arguments for the replayer.
    """

    def __init__(self, args: DataGenAppArgs, env: gym.wrappers.common.OrderEnforcing):
        self.args = args
        self.dataset_file_handler = HDF5DatasetFileHandler()
        self.dataset_file_handler.open(args.hdf5_file)

        self.device = "cuda"
        assert torch.cuda.is_available(), "CUDA is not available"
        episode_count = self.dataset_file_handler.get_num_episodes()
        assert episode_count > 0, "No episodes found in the dataset."

        # Mapper
        self.isaaclab_nvblox_mapper = None
        if includes_nvblox(args.data_type):
            self.isaaclab_nvblox_mapper = IsaacLabNvbloxMapper(
                self.args.data_type, args, self.device
            )

        env_name = self.dataset_file_handler.get_env_name()
        if self.args.task is not None:
            env_name = self.args.task.to_full_task_name()
        assert env_name is not None, "Task/env name was not specified nor found in the dataset."
        embodiment_type = get_embodiment_type_from_task(self.args.task)

        self.env = env

        # Add the Perceptive IL environment helper
        if embodiment_type is EmbodimentType.ARM:
            self.embodiment = ArmEmbodiment(self.args, self.device)
        elif embodiment_type is EmbodimentType.HUMANOID:
            self.embodiment = HumanoidEmbodiment(self.args.task, self.args, self.device)
        else:
            raise ValueError(f"Embodiment type {embodiment_type} not supported.")
        # Initialize the camera handlers
        self.embodiment.initialize_camera_handlers(self.env)
        # Sort demo names numerically by index
        self.available_episode_names = list(self.dataset_file_handler.get_episode_names())
        self.available_episode_names.sort(key=lambda x: int(x.split("_")[1]))

        # Visualizer
        if args.visualize:
            self.visualizer = Visualizer(args)
        else:
            self.visualizer = None

    def simulate_selected_demos(
        self, selected_demo_names: List[str], simulation_app: SimulationAppContext
    ) -> None:
        """Run all selected demonstrations."""

        with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
            while simulation_app.is_running() and not simulation_app.is_exiting():
                for episode_name in selected_demo_names:
                    demo_directory = demo_directory_from_episode_name(
                        self.args.output_dir, episode_name
                    )
                    isaaclab_writer = IsaacLabWriter(demo_directory)
                    outcome_bool = self.try_simulate_episode_max_n_times(
                        episode_name, isaaclab_writer, self.embodiment, self.args.max_num_attempts
                    )
                    outcome_str = "SUCCEEDED" if outcome_bool else "FAILED"
                    print(f"{episode_name} {outcome_str}.")
                    # Save the args for reproducibility.
                    self.args.save(Path(demo_directory) / DATAGEN_ARGUMENT_FILE_NAME)
                break

    def try_simulate_episode_max_n_times(
        self,
        episode_name: str,
        isaaclab_writer: IsaacLabWriter,
        embodiment: EmbodimentBase,
        max_num_attempts: int,
    ) -> bool:
        """Attempt to simulate an episode multiple times until success or max attempts reached.

        Args:
            episode_name (str): Name of the episode to simulate.
            isaaclab_writer (IsaacLabWriter): Writer object for saving episode data.
            embodiment (EmbodimentBase): Helper object for environment.
            max_num_attempts (int): Maximum number of attempts to simulate the episode.

        Returns:
            bool: True if the episode simulation was successful, False otherwise.
        """
        assert (
            episode_name in self.available_episode_names
        ), f"Episode {episode_name} not found in dataset."
        episode_data = self.dataset_file_handler.load_episode(
            episode_name, self.env.unwrapped.device
        ).data

        for retry_idx in range(max_num_attempts):
            print(f"Run {episode_name} attempt {retry_idx + 1} / {max_num_attempts}")
            outcome_bool = self.simulate_episode(episode_data, isaaclab_writer, embodiment)
            if outcome_bool:
                break
        return outcome_bool

    def simulate_episode(
        self,
        episode_data: torch.Tensor,
        isaaclab_writer: IsaacLabWriter,
        embodiment: EmbodimentBase,
    ) -> bool:
        """Simulate a single episode using the provided data.

        Args:
            episode_data (torch.Tensor): Data for the episode to simulate.
            isaaclab_writer (IsaacLabWriter): Writer object for saving episode data.
            embodiment (EmbodimentBase): Helper object for environment.
        Returns:
            bool: True if the episode simulation was successful, False otherwise.
        """
        # Note: Isaclaab modules can only be imported after the simulation app is launched.
        from mindmap.tasks.task_success import get_task_outcome

        # Reset environment to initial state
        self.env.unwrapped.reset_to(episode_data["initial_state"], None, is_relative=True)
        states = episode_data["states"] if "states" in episode_data else None

        actions = episode_data["actions"]
        # Handle different versions of the dataset
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)

        if self.args.task == Tasks.CUBE_STACKING:
            move_up_action = get_move_up_action()
            actions = torch.cat((actions, move_up_action.to(actions.device)), dim=0)

        early_stop = self.args.max_num_steps > 0
        if early_stop:
            actions = actions[: self.args.max_num_steps]

        # Clear the map between episodes
        if self.isaaclab_nvblox_mapper is not None:
            print("Clearing nvblox map")
            self.isaaclab_nvblox_mapper.clear()

        # In tqdm, disable=None means that progress bar is disabled for non-tty sessions (e.g. CI)
        for action_index, action in tqdm.tqdm(enumerate(actions), total=len(actions), disable=None):
            # Initialize sample for visualization
            vis_sample = {
                "vertices": None,
                "vertex_features": None,
            }

            # Step the environment
            action_tensor = torch.Tensor(action)
            if action.dim() == 1:
                action_tensor = action_tensor.reshape([1, action.shape[0]])
            self.env.step(torch.Tensor(action_tensor))

            # Update the nvblox map (skip first frame as sometimes the rgb is greyscale only)
            if includes_nvblox(self.args.data_type) and action_index != 0:
                self.isaaclab_nvblox_mapper.decay()
                for camera_handler in embodiment.camera_handlers:
                    self.isaaclab_nvblox_mapper.update_reconstruction_from_camera(camera_handler)

            # Save the data to disk (skip first frame as sometimes the rgb is greyscale only)
            if self.args.output_dir and action_index != 0:
                # Save end effector states
                state = embodiment.get_robot_state(self.env)
                isaaclab_writer.write_state(state, action_index)
                # Save camera data
                for camera_handler in embodiment.camera_handlers:
                    camera_name = camera_handler.camera_name
                    if includes_rgb(self.args.data_type):
                        isaaclab_writer.write_rgb(
                            camera_handler.get_rgb(), camera_name, action_index
                        )
                    if includes_depth_camera(self.args.data_type):
                        isaaclab_writer.write_depth_camera(camera_handler, action_index)

                # Save nvblox map
                if includes_nvblox(self.args.data_type):
                    vertices, features = self.isaaclab_nvblox_mapper.save_nvblox_map_to_disk(
                        action_index, isaaclab_writer._output_dir
                    )
                    vis_sample["vertices"] = vertices.unsqueeze(0) if vertices is not None else None
                    vis_sample["vertex_features"] = (
                        features.unsqueeze(0).to(torch.float32) if features is not None else None
                    )

            if states and action_index <= len(states) - 1:
                # Iterate over articulation and rigid object states to compare with recorded states
                current_runtime_state = self.env.unwrapped.scene.get_state(is_relative=True)
                states_matched, comparison_log = compare_states(
                    states, current_runtime_state, action_index
                )
                if states_matched:
                    print("\t- matched.")
                else:
                    print("\t- mismatched.")
                    print(comparison_log)

            # Visualizer
            if self.visualizer:
                self.visualizer.visualize(
                    vis_sample,
                    prediction=None,
                    data_type=self.args.data_type,
                    isaac_lab_nvblox_mapper=self.isaaclab_nvblox_mapper,
                )
                self.visualizer.run_until_space_pressed()

        outcome_bool = get_task_outcome(self.args.task, self.env.unwrapped)
        # If the episode was stopped early, we consider it a success.
        success = outcome_bool or early_stop
        if self.args.output_dir:
            outcome = DemoOutcome.SUCCESS if success else DemoOutcome.FAILED_DATAGEN
            isaaclab_writer.write_outcome(outcome)
        return success


def main():
    # Launch the simulator. The context manager ensures that the app is closed also if an error occurs.
    with SimulationAppContext(headless=args.headless, enable_cameras=True) as simulation_app:
        # Note: Isaclaab modules can only be imported after the simulation app is launched.
        from mindmap.isaaclab_utils.environments import SimEnvironment

        with SimEnvironment(args) as sim_environment:
            replayer = IsaacLabDataGenerator(args, sim_environment.env)
            selected_demo_names = get_episode_names(args.demos_datagen)
            replayer.simulate_selected_demos(selected_demo_names, simulation_app)


if __name__ == "__main__":
    main()

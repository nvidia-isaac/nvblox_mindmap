# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import copy
import os
import pathlib
import re
from typing import List, Optional, Tuple, Type

from tap import Tap

from mindmap.closed_loop.closed_loop_mode import ClosedLoopMode
from mindmap.data_loading.data_types import DataType
from mindmap.data_loading.vertex_sampling import VertexSamplingMethod
from mindmap.image_processing.feature_extraction import FeatureExtractorType
from mindmap.isaaclab_utils.render_settings import RenderSettings
from mindmap.keyposes.keypose_detection_mode import KeyposeDetectionMode
from mindmap.tasks.tasks import Tasks

DATAGEN_ARGUMENT_FILE_NAME = "datagen_args.json"
TRAINING_ARGUMENT_FILE_NAME = "training_args.json"
CLOSED_LOOP_ARGUMENT_FILE_NAME = "closed_loop_args.json"


def parse_two_3d_bounds(bounds_str: str) -> Tuple[List[float], List[float]]:
    bounds_str = re.sub(r"[ \[\]]", "", bounds_str)  # Removes spaces, "[" and "]"
    vec = [float(v) for v in bounds_str.split(",")]
    assert len(vec) == 6
    min, max = vec[:3], vec[3:]
    assert min[0] <= max[0] and min[1] <= max[1] and min[2] <= max[2]
    return min, max


#### BASE ARGUMENT CLASSES ####
# Argument classes that are common in multiple applications.
# Should not be used directly, but only as base classes for application specific arguments classes.


class ModelArgs(Tap):
    # Model input data params
    use_keyposes: int = 1  # Predict next keypose instead of a dense trajectory.
    # These two keypose parameters will be set to task defaults if not explicitly specified.
    extra_keyposes_around_grasp_events: Optional[List[int]] = None
    keypose_detection_mode: Optional[KeyposeDetectionMode] = None
    add_external_cam: bool = False  # Add an external camera to the ego-camera.
    gripper_encoding_mode: str = (
        "binary"  # Only applies to non-keypose mode. One of ['binary', 'analog']
    )
    only_sample_keyposes: bool = False
    # Model constructor arguments
    image_size: Tuple[int, int] = (512, 512)
    feature_image_size: Tuple[int, int] = (32, 32)
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 0
    fps_subsampling_factor: int = 5
    use_fps: int = 1
    rotation_parametrization: str = "6D_from_query"
    quaternion_format: str = "wxyz"
    diffusion_timesteps: int = 100
    num_history: int = 3  # Number of previous samples to use as input for prediction
    prediction_horizon: int = 1  # Number of samples to predict.
    relative_action: int = 0
    lang_enhanced: int = 0
    # Encodings
    data_type: DataType = DataType.RGBD_AND_MESH
    encode_openness: int = 1
    feature_type: FeatureExtractorType = FeatureExtractorType.RADIO_V25_B
    use_shared_feature_encoder: int = 0  # Features from both mesh and rgb are encoded to embedding_dim in the model. This parameter governs whether the same (shared) encoder is used for both of them or if individual encoders should be used.
    # Sampling-based params
    vertex_sampling_method: VertexSamplingMethod = VertexSamplingMethod.RANDOM_WITHOUT_REPLACEMENT
    num_vertices_to_sample: int = 2048
    rgbd_min_depth_threshold: float = 0.0  # Only include camera depth values above this value.
    # Loss weights
    pos_loss: float = 30.0
    rot_loss: float = 10.0
    gripper_loss: float = 1.0
    # Data augmentation and regularization
    apply_random_transforms: int = 0  # Whether to apply random transformations
    apply_geometry_noise: int = 0  # Whether to apply random noise to poses and vertices
    pos_noise_stddev_m: float = (
        0.01  # Standard deviation of noise applied to 3d points and camera pos
    )
    rot_noise_stddev_deg: float = 0.01  # Standard deviation of rotation noise added to poses
    encoder_dropout: float = 0.0
    diffusion_dropout: float = 0.0
    predictor_dropout: float = 0.0
    # Task related params
    task: Optional[Tasks] = None

    def configure(self):
        self.add_argument(
            "--random_translation_range_m",
            type=parse_two_3d_bounds,
            default="[[-0.1, -0.1, 0.0], [0.1, 0.1, 0.0]]",
        )
        self.add_argument(
            "--random_rpy_range_deg",
            type=parse_two_3d_bounds,
            default="[[0.0, 0.0, -90.0], [0.0, 0.0, 90.0]]",
        )


class DataGenArgs(Tap):
    include_dynamic: bool = False  # map the dynamic (the robot arm) voxels as well
    validate_demos_with_gt_poses: int = 1  # Whether to reject demos where the GT-policy failed.
    voxel_size_m: float = None  # Override default value from nvblox_mapper_constants.py
    projective_appearance_integrator_measurement_weight: float = (
        None  # Override default value from nvblox_mapper_constants.py
    )
    demos_datagen: str = (
        "0"  # A list of demo indices to be replayed (can handle ranges e.g. --demos 1 2 3-5 7).
    )
    save_serialized_nvblox_map_to_disk: bool = (
        False  # Write the nvblox map to the output directory.
    )


class ClosedLoopArgs(Tap):
    # Environment params
    demos_closed_loop: str = (
        "0"  # A list of demo indices to be replayed (can handle ranges e.g. --demos 1 2 3-5 7).
    )
    num_retries: int = 1  # Number of retries for each demo.
    demo_mode: ClosedLoopMode = ClosedLoopMode.CLOSED_LOOP_WAIT  # Mode to run this script in.
    max_num_steps_to_goal: int = 40
    # Controller params
    terminate_after_n_steps: int = None  # Terminate the simulation after this many steps if passed.
    max_intermediate_distance_m: float = None  # Maximum distance between intermediate goals.
    # Logging and evaluation params
    eval_file_path: str = None  # Path to store the .json evaluation file.
    # Recording camera params
    record_camera_output_path: str = None  # The path to store recorded camera images and videos
    record_videos: bool = False
    video_size: Tuple[int, int] = (320, 320)  # Image size of recorded video
    # GT predictions mode params
    gt_goals_subsampling_factor: int = 5  # only used in execute_gt_goals + use_keyposes = 0 mode


class SystemArgs(Tap):
    # General params
    seed: int = 0
    ignore_model_args_json: bool = False
    # Path params
    checkpoint: Optional[pathlib.Path] = None
    fpn_checkpoint: Optional[pathlib.Path] = None
    dataset: str = None  # Path to the (validation) dataset. Used in keypose execution mode to get the keyposes.
    base_log_dir: pathlib.Path = "/eval/train_logs/"
    # Logging params
    wandb_name: str = None
    wandb_mode: str = "online"  # One of [online, offline, disabled]
    wandb_entity: str = "nv-welcome"


class SimulationArgs(Tap):
    headless: bool = False  # Whether to run IsaacLab in headless mode.
    num_envs: int = 1  # Number of environments to simulate.
    hdf5_file: str = None  # Path to the hdf5 file.
    background_env_usd_path: str = None  # Path to the USD file of the background env.
    render_settings: RenderSettings = RenderSettings.DEFAULT
    sim_device: str = (
        "cpu"  # Device used for simulation (cpu or cuda). CPU is known to be more stable.
    )
    verbose: bool = False  # Propagated to isaac lab
    disable_fabric: bool = False


class VisualizerArgs(Tap):
    visualize_backprojected_features: bool = False
    visualize_encoded_features: bool = True
    visualize_attention_weights: bool = False
    visualize_aabb: bool = False
    visualizer_record_camera_output_path: str = None  # The path to store recording camera outputs.
    visualizer_voxel_size_m: float = 0.01
    # FIXME(cvolk): This is set wrt. world reference frame and does not work for the stick_in_bin task.
    visualizer_pointcloud_max_distance: float = None
    visualizer_min_tsdf_weight: float = 0.0
    visualizer_point_size: int = 3  # Point size for the pointclouds
    disable_visualizer_wait_on_key: bool = False
    visualize: bool = False
    visualizer_background_rgb: float = (0.0, 0.0, 0.0)  # Range: [0.0, 1.0]
    visualizer_pointclouds_ply_output_dir: str = None  # The path to store pointclouds as ply files.
    visualizer_min_attention_weight: float = 0.0


#### APPLICATION SPECIFIC ARGUMENT CLASSES ####
# Argument classes that are specific to an application.
# Pulling in common arguments from the base classes.


class DataGenAppArgs(ModelArgs, SimulationArgs, SystemArgs, VisualizerArgs, DataGenArgs):
    output_dir: str = None  # Directory to store output files.
    add_depth_noise: bool = False  # Add noise to the depth camera output.
    max_num_attempts: int = 5  # Maximum number of attempts until a demo is successful
    max_num_steps: int = -1  # Max number of steps to run the simulation for (-1 for no limit).

    def process_args(self):
        if self.add_external_cam and self.data_type == DataType.RGBD_AND_MESH:
            raise ValueError("RGBD_AND_MESH data type has only been tested with ego-cam")


class OpenLoopAppArgs(ModelArgs, SystemArgs, VisualizerArgs):
    demos_open_loop: str = (
        "0"  # A list of demo indices to be replayed (can handle ranges e.g. --demos 1 2 3-5 7).
    )
    pass


class ClosedLoopAppArgs(
    ModelArgs, SimulationArgs, SystemArgs, VisualizerArgs, DataGenArgs, ClosedLoopArgs
):
    visualize_robot_state: bool = False

    def process_args(self):
        assert self.prediction_horizon == 1 or self.demo_mode != ClosedLoopMode.EXECUTE_GT_GOALS


class ValidateDemosAppArgs(SimulationArgs, SystemArgs, ClosedLoopArgs):
    pass


class TrainingAppArgs(ModelArgs, SystemArgs, VisualizerArgs, DataGenArgs):
    max_episodes_per_task: int = 100
    instructions: Optional[pathlib.Path] = None
    variations: Tuple[int, ...] = (0,)
    eval_only: bool = False
    save_checkpoint: bool = True
    # Training and validation datasets
    demos_train: str = (
        "0"  # A list of demo indices to be replayed (can handle ranges e.g. --demos 1 2 3-5 7).
    )
    demos_valset: Optional[str] = None
    include_failed_demos: bool = False
    # Logging
    exp_name: str = "mindmap Training"
    # Main training parameters
    num_workers: int = 0
    num_workers_for_test_dataset: Optional[
        int
    ] = None  # None or <0 means use the same number of workers as the train loader.
    batch_size: int = 32
    batch_size_val: int = 32
    initial_learning_rate: float = 1e-4
    learning_rate_end_factor: float = 0.5  # 1.0 means no learning rate decay
    learning_rate_convergence_percentage: float = 0.75
    train_iters: int = 100000
    accumulate_grad_batches: int = 1
    val_freq: int = 100
    print_timers_freq: int = 1000  # used for print timers to console every N training iterations
    print_progress_freq = 100  # used for print progress to console every N training iterations
    num_batches_per_train_eval: int = 10  # -1 for full dataset
    num_batches_per_test_eval: int = -1  # -1 for full dataset
    max_episode_length: int = 5  # -1 for no limit
    viz_freq: int = 200  # visualize on wandb every N iterations
    skip_train_val: bool = False
    sampling_weighting_type: str = "uniform"  # ['uniform', 'gripper_state_change']

    def process_args(self):
        if self.add_external_cam and self.data_type == DataType.RGBD_AND_MESH:
            raise ValueError("RGBD_AND_MESH data type has only been tested with ego-cam")


def print_arg_differences(initial_args: Tap, updated_args: Tap) -> None:
    """Print the differences between the initial and updated arguments."""
    # Get the union of keys from both dictionaries
    initial_args_dict = initial_args.as_dict()
    updated_args_dict = updated_args.as_dict()
    assert initial_args_dict.keys() == updated_args_dict.keys()
    for key in initial_args_dict:
        value1 = initial_args_dict[key]
        value2 = updated_args_dict[key]
        if value1 != value2:
            print(f"Updated Argument: {key}")
            print(f"  Initial: {value1}")
            print(f"  Updated: {value2}")


def extract_args_belonging_to_class(arguments: Tap, cls: Type[Tap]) -> dict:
    """
    Extracts the arguments from the given `arguments` that belong to the `cls` class.

    Args:
        arguments (Tap): The arguments to extract from.
        cls (Type[Tap]): The class to filter the arguments by.

    Returns:
        dict: A dictionary containing the arguments that belong to the `cls` class.
    """
    # All argument keys that belong to the ModelArgs class
    model_args_keys = cls().parse_args([]).as_dict().keys()
    # The dictionary of arguments that should be filtered for ModelArgs.
    args_dict = arguments.as_dict()
    # Remove all key/value pairs that do not belong to the ModelArgs class
    return {k: v for k, v in args_dict.items() if k in model_args_keys}


def update_model_args_from_checkpoint(cli_args: Tap):
    """
    If available, load the model arguments from a JSON file (belonging to the selected checkpoint)
    and overwrite the model arguments coming from CLI with the loaded arguments.

    Args:
        cli_args (Tap): The command line arguments.

    Returns:
        Tap: The updated command line arguments.

    Notes:
        - We only overwrite the arguments part of the ModelArgs class.
          - i.e. when the cli_args are of type TrainingAppArgs,
            only the arguments inherited from ModelArgs are overwritten.
          - This way, we keep the flexibility to run a model with different train/evaluation arguments
            while ensuring that we don't initiate the model differently than the checkpoint was trained.
    """
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", -1)))
    # Load model args from json file.
    if cli_args.checkpoint:
        if not cli_args.ignore_model_args_json:
            args_file_path = cli_args.checkpoint.parent / TRAINING_ARGUMENT_FILE_NAME
            if args_file_path.is_file():
                print(f"Loading model args from {args_file_path}")
                # Load the arguments stored in the json file.
                loaded_args = Tap().load(args_file_path)
                # Extract only the model arguments.
                # NOTE(remos): This is needed to not overwrite any other args than the model args.
                loaded_model_args = extract_args_belonging_to_class(loaded_args, ModelArgs)
                # Overwrite the model arguments coming from CLI with the once from json.
                updated_args = copy.deepcopy(cli_args)
                updated_args.from_dict(loaded_model_args)
                if rank <= 0:
                    print(
                        f"Loaded model args from {args_file_path}. Complete args list:\n{updated_args}"
                    )
                    print_arg_differences(cli_args, updated_args)
                return updated_args
            else:
                print(f"Requested model args path {args_file_path} does not exist.")
        else:
            print("Loading checkpoint without loading model args. Danger Will Robinson!")
    else:
        print("No checkpoint provided.")

    if rank == 0:
        print(
            f"Not loading model args from file, only using CLI args. Complete args list:\n{cli_args}"
        )
    return cli_args

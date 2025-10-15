# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import glob
import gzip
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

from catalyst.data.sampler import DistributedSamplerWrapper
import imageio.v2 as imageio
import numpy as np
from nvblox_torch.timer import Timer
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import zstandard

from mindmap.common_utils.demo_selection import get_demo_paths
from mindmap.data_loading.batching import collate_batch
from mindmap.data_loading.data_types import DataType
from mindmap.data_loading.item_names import (
    GT_POLICY_STATE_PRED_ITEM_NAME,
    POLICY_STATE_HISTORY_ITEM_NAME,
)
from mindmap.data_loading.sample_transformer import SampleTransformer
from mindmap.data_loading.sampling_weighting_type import SamplingWeightingType
from mindmap.embodiments.embodiment_base import EmbodimentBase
from mindmap.embodiments.state_base import PolicyStateBase, RobotStateBase
from mindmap.isaaclab_utils.isaaclab_datagen_utils import DemoOutcome
from mindmap.keyposes.keypose_detection_mode import KeyposeDetectionMode
from mindmap.keyposes.task_to_default_keypose_params import (
    TASK_TYPE_TO_EXTRA_KEYPOSES_AROUND_GRASP_EVENTS,
    TASK_TYPE_TO_KEYPOSE_DETECTION_MODE,
)
from mindmap.model_utils.distributed_training import print_dist
from mindmap.tasks.tasks import Tasks


class IsaacLabDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        demos: str,
        task: Tasks,
        embodiment: EmbodimentBase,
        item_names: List[str],
        transforms: Dict = {},
        only_sample_keyposes: bool = False,
        include_failed_demos: bool = False,
        num_history: int = 3,
        prediction_horizon: int = 1,
        use_keyposes: bool = False,
        extra_keyposes_around_grasp_events: Optional[List[int]] = None,
        keypose_detection_mode: Optional[KeyposeDetectionMode] = None,
        gripper_encoding_mode: str = "binary",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes the IsaacLabDataset.

        Args:
            dataset_path (str): The path to the dataset (folder containing the demo directories).
            demos (str): The demos indices to load represented as a range string e.g. "0-5 7 9-11".
            item_names (list[str]): The names of the items in the dataset.
                - E.g to load <idx>.table_rgb.png and <idx>.table_pcd.npy: ['table_rgb.png', 'table_pcd.npy']
            transforms (Dict[str, callable]): The transformations to apply to each item.
                - E.g to transform the <idx>.table_rgb.png images: {'table_rgb.png', transform_image}
            only_sample_keyposes (bool, optional): Whether to only sample keyposes.
            include_failed_demos (bool, optional): Whether to also load failed demos (indicated by 'demo_successful.npy').
            dtype (torch.dtype, optional): The data type of the tensors. Defaults to torch.float32.
        """

        self.item_names = item_names
        self.transforms = transforms
        self.only_sample_keyposes = only_sample_keyposes
        self.dtype = dtype

        self.gripper_encoding_mode = gripper_encoding_mode
        self.num_history = num_history
        self.prediction_horizon = prediction_horizon
        self.use_keyposes = use_keyposes
        self.excluded_datasets = 0
        self.sample_paths = {}
        self.dataset_path_list = get_demo_paths(dataset_path, demos)

        # Task specific extra keyposes
        if extra_keyposes_around_grasp_events is None:
            extra_keyposes_around_grasp_events = TASK_TYPE_TO_EXTRA_KEYPOSES_AROUND_GRASP_EVENTS[
                task.name
            ]
        if keypose_detection_mode is None:
            keypose_detection_mode = TASK_TYPE_TO_KEYPOSE_DETECTION_MODE[task.name]

        self.embodiment = embodiment

        for dataset_path in self.dataset_path_list:
            assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist."

            if not include_failed_demos and not self.is_demo_successful(dataset_path):
                self.excluded_datasets += 1
                continue

            self.sample_paths[dataset_path] = {}

            # Load the policy states and keypose indices from the disk
            policy_states, keypose_indices = self.load_policy_states_and_keyposes(
                dataset_path=dataset_path,
                embodiment=self.embodiment,
                extra_keyposes_around_grasp_events=extra_keyposes_around_grasp_events,
                keypose_detection_mode=keypose_detection_mode,
                use_keyposes=self.use_keyposes,
            )

            # Cache the policy states and keypose indices for later retrieval
            self.sample_paths[dataset_path]["policy_states"] = policy_states
            self.sample_paths[dataset_path]["keypose_indices"] = keypose_indices

            num_samples_in_dataset = len(policy_states)
            for item_name in item_names:
                # Ignore items that are computed on the fly
                if item_name.startswith("runtime_"):
                    continue
                # Find all samples of the given item.
                item_path = os.path.join(dataset_path, "*." + item_name)
                sample_paths = glob.glob(item_path)
                assert len(sample_paths) > 0, f"No samples found in {item_path}."
                sample_paths = sorted(
                    sample_paths, key=lambda x: int(os.path.basename(x).split(".")[0])
                )
                is_keypose_list = np.full(len(sample_paths), False)
                is_keypose_list[self.sample_paths[dataset_path]["keypose_indices"]] = True

                if self.only_sample_keyposes:
                    # Filter out non-keypose samples.
                    sample_paths = [
                        path
                        for path, is_keypose in zip(sample_paths, is_keypose_list)
                        if is_keypose
                    ]
                self.sample_paths[dataset_path][item_name] = sample_paths
                assert (
                    len(sample_paths) == num_samples_in_dataset
                ), f"Found {len(sample_paths)} samples of {item_name} in {dataset_path}, but should have {num_samples_in_dataset}."
            self.sample_paths[dataset_path]["num_samples"] = num_samples_in_dataset
        self.total_number_of_samples = sum(
            [dataset["num_samples"] for dataset in self.sample_paths.values()]
        )
        # Update the path list with only the non-excluded demos.
        self.dataset_path_list = self.sample_paths.keys()
        print_dist(
            f"Found {self.total_number_of_samples} samples from "
            f"{len(self.dataset_path_list)} datasets ({self.excluded_datasets} datasets have been excluded)."
        )
        print_dist(f"Dataset items: {self.item_names}")

    def is_demo_successful(self, dataset_path: str) -> bool:
        """
        Check if the demo in the given dataset path is successful.
        This is done by checking a flag in the 'demo_successful.npy' file.

        Args:
            dataset_path (str): The path to the dataset.
        """
        is_successful_file = os.path.join(dataset_path, "demo_successful.npy")
        demo_outcome = DemoOutcome(np.load(is_successful_file))
        return demo_outcome == DemoOutcome.SUCCESS

    def get_sample_weights(
        self, sampling_weighting_type: SamplingWeightingType, use_keyposes: bool
    ) -> np.ndarray:
        """
        Returns an array of sample weights based on the given sampling weighting type.
        """
        if sampling_weighting_type == SamplingWeightingType.UNIFORM:
            return np.ones(self.total_number_of_samples)
        elif sampling_weighting_type == SamplingWeightingType.GRIPPER_STATE_CHANGE:
            return self._get_gripper_state_change_weights(use_keyposes)
        else:
            print_dist(f"Unknown sampling weighting type: {sampling_weighting_type}")
            raise NotImplementedError

    def _get_gripper_state_change_weights(self, use_keyposes: bool) -> np.ndarray:
        """
        Calculates sample weights based on whether the gripper state changes.
        """
        # Iterate through all samples and check whether the gripper state changes.
        has_state_change_in_samples = np.empty(self.total_number_of_samples, dtype=bool)
        for global_idx in range(self.total_number_of_samples):
            dataset_path, sample_idx = self.get_dataset_sample_from_global_idx(global_idx)
            if use_keyposes:
                has_state_change_in_samples[
                    global_idx
                ] = self._keypose_sample_has_gripper_state_change(dataset_path, sample_idx)
            else:
                has_state_change_in_samples[
                    global_idx
                ] = self._trajectory_sample_has_gripper_state_change(dataset_path, sample_idx)

        # Either there is a state change or not
        state_change_classes = [False, True]
        # Get array with [number of samples without state change, number of samples with state changes]
        state_change_class_counts = np.array(
            [
                len(np.where(has_state_change_in_samples == state_change_class)[0])
                for state_change_class in state_change_classes
            ]
        )
        # Get array with [weight of samples without state change, weight of samples with state changes]
        assert np.all(state_change_class_counts != 0), "Found no samples in at least one class."
        state_change_class_weights = 1.0 / state_change_class_counts
        # Get array with weights for each sample
        sample_weights = np.array(
            [
                state_change_class_weights[int(has_state_change_in_sample)]
                for has_state_change_in_sample in has_state_change_in_samples
            ]
        )
        assert len(sample_weights) == self.total_number_of_samples
        return sample_weights

    def _keypose_sample_has_gripper_state_change(self, dataset_path: str, sample_idx: int) -> bool:
        """
        Checks if the gripper state changes from the previous to the next keypose.
        """
        # Load the gripper states in the history and prediction of the sample.
        previous_keyposes = np.load(
            self.sample_paths[dataset_path]["previous_keyposes.npy"][sample_idx]
        )
        next_keyposes = np.load(self.sample_paths[dataset_path]["next_keyposes.npy"][sample_idx])
        # Check if the previous gripper state is the same as the next gripper state.
        return previous_keyposes[-1, -1] != next_keyposes[0, -1]

    def _trajectory_sample_has_gripper_state_change(
        self, dataset_path: str, sample_idx: int
    ) -> bool:
        """
        Checks if the gripper state changes in any of the poses in the past + future trajectory.
        """
        # Load the gripper states in the history and prediction of the sample.
        gripper_history = np.load(
            self.sample_paths[dataset_path]["gripper_history.npy"][sample_idx]
        )
        gt_gripper_pred = np.load(
            self.sample_paths[dataset_path]["gt_gripper_pred.npy"][sample_idx]
        )
        gripper_openness = np.concatenate((gripper_history[:, -1], gt_gripper_pred[:, -1]))
        # Check whether the gripper state changes i.e. the number of unique states is > 1
        number_of_unique_gripper_states = len(np.unique(gripper_openness))
        return number_of_unique_gripper_states > 1

    def get_dataset_sample_from_global_idx(self, global_idx: int) -> Tuple[str, int]:
        """
        Retrieves the path of the dataset and sample index corresponding to a global index.
            - E.g. if dataset1 is of size 100, dataset2 is of size 200:
                - global_idx = 99 -> dataset1_path, 99
                - global_idx = 100 -> dataset2_path, 0

        Args:
            global_idx (int): The global (inter-dataset) index of the sample.

        Returns:
            Tuple[str, int]: A tuple containing the dataset path and the starting index of the dataset.
        """
        dataset_start_idx = 0
        next_dataset_start_idx = 0
        for dataset_path in self.dataset_path_list:
            next_dataset_start_idx += self.sample_paths[dataset_path]["num_samples"]
            if global_idx < next_dataset_start_idx:
                break
            dataset_start_idx = next_dataset_start_idx
            assert dataset_start_idx < len(self)
        sample_idx = global_idx - dataset_start_idx
        return dataset_path, sample_idx

    def __len__(self):
        return self.total_number_of_samples

    def get_policy_state_history(
        self, sample_idx: int, candidate_indices: np.array, gripper_states: List[RobotStateBase]
    ):
        """Get self.num_history poses up to (and including) sample_idx. If there are not enough
        poses before sample_idx, the first pose will be repeated.

        Args:
            sample_idx   Index for which history is to be retreived
            candidate_indices Indices that should be considered, either keypose indices or all indices.
        """

        # Get indices up to and including sample_idx
        history_indices = candidate_indices[candidate_indices <= sample_idx][-self.num_history :]

        # If we don't have enough indices, we're filling up with index 0 at the beginning
        num_missing = self.num_history - history_indices.shape[0]
        if num_missing > 0:
            history_indices = np.concatenate((np.zeros(num_missing, dtype=int), history_indices))

        gripper_history = [gripper_states[i] for i in history_indices]
        assert len(gripper_history) == self.num_history

        return gripper_history

    def get_policy_state_future(
        self, sample_idx: int, candidate_indices: np.array, gripper_states: List[RobotStateBase]
    ):
        """Get self.prediciont_horizon poses after sample_idx. If there are not enough
        poses after sample_idx, the last pose will be repeated.

        Args:
            sample_idx   Index for which history is to be retrieved
            candidate_indices Indices that should be considered, either keypose indices or all indices.
        """

        # Get indices up to and including sample_idx
        future_indices = candidate_indices[candidate_indices > sample_idx][
            : self.prediction_horizon
        ]

        # If we don't have enough indices, we're filling with the last index at the end
        num_missing = self.prediction_horizon - future_indices.shape[0]
        if num_missing > 0:
            future_indices = np.concatenate(
                (future_indices, np.full(num_missing, fill_value=candidate_indices[-1], dtype=int))
            )

        gripper_future = [gripper_states[i] for i in future_indices]

        assert len(gripper_future) == self.prediction_horizon

        return gripper_future

    def load_robot_states(
        self, dataset_path: str, embodiment: EmbodimentBase
    ) -> List[RobotStateBase]:
        """Load robot states from disk.

        Args:
            dataset_path Path to load from
        """
        robot_state_files = sorted(glob.glob(os.path.join(dataset_path, "*." + "robot_state.npy")))
        if len(robot_state_files) == 0:
            # TODO(remos): Remove this once we have updated all the data.
            print(
                f"WARNING: No robot state files found named *.robot_state.npy."
                "Assuming this is old with robot states named *.gripper_state.npy."
            )
            robot_state_files = sorted(
                glob.glob(os.path.join(dataset_path, "*." + "gripper_state.npy"))
            )
            if len(robot_state_files) == 0:
                raise ValueError(f"No robot state files found in {dataset_path}")
        robot_states: List[RobotStateBase] = []
        for i, path in enumerate(robot_state_files):
            robot_state_tensor = torch.from_numpy(np.load(path, allow_pickle=True))
            robot_states.append(embodiment.robot_state_type.from_tensor(robot_state_tensor))

        return robot_states

    def load_policy_states_and_keyposes(
        self,
        dataset_path: str,
        embodiment: EmbodimentBase,
        extra_keyposes_around_grasp_events: List[int],
        keypose_detection_mode: KeyposeDetectionMode,
        use_keyposes: bool,
    ) -> Tuple[List[PolicyStateBase], List[int]]:
        """Load policy states and keyposes.

        This function loads the robot states from the disk and extracts the keypose indices.
        It then converts the robot states to policy states, which are used in training.

        Args:
            dataset_path (str): The path to the dataset.
            embodiment (EmbodimentBase): The embodiment to use.
            extra_keyposes_around_grasp_events (List[int]): The number of keyposes to sample around
                grasp events.
            keypose_detection_mode (KeyposeDetectionMode): The mode to use for keypose detection.
            use_keyposes (bool): Whether to use keyposes.

        Returns:
            Tuple[List[PolicyStateBase], List[int]]:
                - policy_states: The policy states.
                - keypose_indices: The keypose indices.
        """
        # Load the robot states from the disk
        robot_states: List[RobotStateBase] = self.load_robot_states(dataset_path, embodiment)

        # Extract the keypose indices
        keypose_indices = self.embodiment.keypose_estimator.extract_keypose_indices(
            robot_states, extra_keyposes_around_grasp_events, keypose_detection_mode
        )

        # Convert the robot states to policy states
        policy_states = self.embodiment.offline_estimator.policy_states_from_robot_states(
            robot_states, use_keyposes
        )

        # If we're only using keyposes, we filter out all other samples
        if self.only_sample_keyposes:
            policy_states = [policy_states[i] for i in keypose_indices]
            assert len(policy_states) == len(keypose_indices)

        return policy_states, keypose_indices

    def unpickle_zst(self, item_path):
        dctx = zstandard.ZstdDecompressor()
        with open(item_path, "rb") as infile:
            with dctx.stream_reader(infile) as reader:
                sample = pickle.load(reader)
        return sample

    def unpickle_gz(self, item_path):
        with gzip.open(item_path, "rb") as f:
            sample = pickle.load(f)
        return sample

    def __getitem__(self, global_idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Args:
            global_idx (int): The global index (inter-dataset) of the sample to retrieve.

        Returns:
            dict: A dictionary of tensors and dicts, each representing a sample of a specific item from the dataset.
        """
        timer = Timer("data_engine/getitem")
        assert global_idx < self.total_number_of_samples
        dataset_path, sample_idx = self.get_dataset_sample_from_global_idx(global_idx)

        # Get the  gripper states and keyposes
        policy_states = self.sample_paths[dataset_path]["policy_states"]
        keypose_indices = self.sample_paths[dataset_path]["keypose_indices"]
        num_samples_in_demo = self.sample_paths[dataset_path]["num_samples"]
        assert len(policy_states) == num_samples_in_demo
        assert len(keypose_indices) > 0

        # Determine which sample indices to use for retrieving history and future
        if self.use_keyposes:
            if self.only_sample_keyposes:
                candidate_indices = np.arange(0, len(keypose_indices))
            else:
                candidate_indices = keypose_indices
        else:
            candidate_indices = np.arange(0, num_samples_in_demo)

        # Some of the transforms contains internal state (e.g. data augmentation should apply the
        # same transform to all items in a sample). We therefor need to reset them beforehand.
        for transforms in self.transforms.values():
            for t in transforms:
                t.reset()

        # Load a sample of each item.
        samples = {}
        for item_name in self.item_names:
            extension = os.path.basename(item_name).split(".")[-1]
            if extension == "npy":
                item_path = self.sample_paths[dataset_path][item_name][sample_idx]
                sample = torch.as_tensor(np.load(item_path)).to(self.dtype)
            elif extension == "png":
                item_path = self.sample_paths[dataset_path][item_name][sample_idx]
                sample = torch.as_tensor(imageio.imread(item_path)).to(self.dtype)
            elif extension == "zst":
                item_path = self.sample_paths[dataset_path][item_name][sample_idx]
                sample = self.unpickle_zst(item_path)
            elif item_name == POLICY_STATE_HISTORY_ITEM_NAME:
                gripper_history = self.get_policy_state_history(
                    sample_idx, candidate_indices, policy_states
                )
                sample = gripper_history
            elif item_name == GT_POLICY_STATE_PRED_ITEM_NAME:
                gripper_future = self.get_policy_state_future(
                    sample_idx, candidate_indices, policy_states
                )
                sample = gripper_future
            elif item_name == "runtime_is_keypose":
                sample = (
                    torch.tensor(True)
                    if self.only_sample_keyposes
                    else torch.tensor(sample_idx in keypose_indices)
                )
            else:
                raise ValueError(f"Unsupported item: {item_name}")

            # Apply a transform if given for the item.
            if item_name in self.transforms:
                for transform in self.transforms[item_name]:
                    sample = transform(sample)

            samples[item_name] = sample

        timer.stop()
        return samples


def get_dataloader(
    dataset_path: str,
    demos: str,
    task: Tasks,
    embodiment: EmbodimentBase,
    item_names: List[str],
    transforms: Dict[str, List[SampleTransformer]],
    num_workers: int,
    batch_size: int,
    use_keyposes: bool,
    data_type: DataType,
    only_sample_keyposes: bool,
    extra_keyposes_around_grasp_events: List[int],
    keypose_detection_mode: KeyposeDetectionMode,
    include_failed_demos: bool,
    sampling_weighting_type: SamplingWeightingType,
    gripper_encoding_mode: str,
    num_history: int,
    prediction_horizon: int,
    seed: int = 0,
) -> Tuple[DataLoader, WeightedRandomSampler]:
    """
    Creates a DataLoader for the IsaacLabDataset.

    Args:
        dataset_path (str): The path to the dataset.
        demos (str): The demos indices to load represented as a range string e.g. "0-5 7 9-11".
        item_names (List[str]): The names of the items to load.
        transforms (List[str]): The transforms to apply to the items.
        num_workers (int): The number of worker processes.
        batch_size (int): The number of samples in a batch.
        use_keyposes (bool): Whether to load sparsely detected keyposes or dense trajectories into
                             gripper_history and gt_gripper_pred.
        data_type (DataType, optional): The type of data to load.
        only_sample_keyposes (bool): Whether to only sample keyposes.
        extra_keyposes_around_grasp_events (List[int]): The number of keyposes to sample around
            grasp events.
        keypose_detection_mode (KeyposeDetectionMode): The mode to use for keypose detection.
        include_failed_demos (bool): Whether to also load failed demos (indicated by 'demo_successful.npy').
        sampling_weighting_type (SamplingWeightingType): The type of sampling weighting to use.
        seed (int): random seed used in sampler shuffling.
    Returns:
        DataLoader: The DataLoader for the IsaacLabDataset.
        Sampler: DistributedSampler if dist available and init, otherwise None
    """
    assert (
        use_keyposes or not only_sample_keyposes
    ), "only_sample_keyposes only works with use_keyposes"

    # Create an instance of the dataset.
    dataset = IsaacLabDataset(
        dataset_path,
        demos=demos,
        task=task,
        embodiment=embodiment,
        item_names=item_names,
        transforms=transforms,
        only_sample_keyposes=only_sample_keyposes,
        include_failed_demos=include_failed_demos,
        use_keyposes=use_keyposes,
        keypose_detection_mode=keypose_detection_mode,
        extra_keyposes_around_grasp_events=extra_keyposes_around_grasp_events,
        gripper_encoding_mode=gripper_encoding_mode,
        num_history=num_history,
        prediction_horizon=prediction_horizon,
    )

    distributed_weighted_sampler = None
    if sampling_weighting_type != SamplingWeightingType.NONE:
        # Handle data imbalance by using a weighted random sampler
        generator = torch.Generator()
        generator.manual_seed(seed)
        # When running with uniform sampling, we don't want replacement (i.e. all samples will be sampled exactly once)
        # When running with weighted sampling, we want replacement (i.e. some samples may be sampled more than once)
        replacement = False if sampling_weighting_type == SamplingWeightingType.UNIFORM else True
        weights = dataset.get_sample_weights(sampling_weighting_type, use_keyposes)
        weighted_sampler = WeightedRandomSampler(
            weights, len(dataset), generator=generator, replacement=replacement
        )
        # Sampler will shuffle the indices, Drop the tail of the data to make it
        # evenly divisible.
        distributed_weighted_sampler = DistributedSamplerWrapper(
            sampler=weighted_sampler, shuffle=True
        )

    # Create a DataLoader instance.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # when using sampler this needs to be False
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_batch,
        sampler=distributed_weighted_sampler,
    )

    return dataloader, distributed_weighted_sampler

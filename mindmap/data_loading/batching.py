# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import Dict, List, Tuple

from nvblox_torch.timer import Timer
import torch

from mindmap.data_loading.data_types import (
    DataType,
    includes_mesh,
    includes_pcd,
    includes_policy_states,
    includes_rgb,
)
from mindmap.data_loading.item_names import (
    GT_POLICY_STATE_PRED_ITEM_NAME,
    IS_KEYPOSE_ITEM_NAME,
    NVBLOX_VERTEX_FEATURES_ITEM_NAME,
    POLICY_STATE_HISTORY_ITEM_NAME,
)
from mindmap.embodiments.embodiment_base import EmbodimentBase, EmbodimentType
from mindmap.embodiments.state_base import state_tensor_from_history_list
from mindmap.image_processing.backprojection import get_camera_pointcloud
from mindmap.image_processing.feature_extraction import FeatureExtractorType


def batch_mesh_vertices(
    mesh_vertices_list: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Create a batched mesh vertices dict.

    Args:
        mesh_vertices_list (List[Dict[str, torch.Tensor]]):  A list of dictionaries,
            where each dictionary contains the components of a featurized mesh vertice pointcloud.

    Returns:
        batched_vertices Dict[str, torch.Tensor]: A dictionary containing the stacked
            version of the featurized mesh vertice pointcloud list.
    """
    batched_vertices = {}

    # Stack feature and vertices.
    batched_vertices["features"] = torch.stack(
        [tensor["features"] for tensor in mesh_vertices_list]
    )
    batched_vertices["vertices"] = torch.stack(
        [tensor["vertices"] for tensor in mesh_vertices_list]
    )
    batched_vertices["vertices_valid_mask"] = torch.stack(
        [tensor["vertices_valid_mask"] for tensor in mesh_vertices_list]
    )

    # These fields should be the same for all samples.
    batched_vertices["channel_length"] = mesh_vertices_list[0]["channel_length"]

    for batch_idx, vertices_dict in enumerate(mesh_vertices_list):
        if batch_idx == 0:
            continue
        assert vertices_dict["channel_length"] == batched_vertices["channel_length"]

    return batched_vertices


def collate_batch(
    data: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collate the data into a batch.

    Args:
        data (List[Dict]): A list of dictionaries, where each dictionary contains all items from a sample.
            - I.e. data = [{"item1": Sample1Item1, "item2": Sample1Item2, ...},
                          {"item1": Sample2Item1, "item2": Sample2Item2, ...}, ...]
    Returns:
        Dict: A dictionary of tensors, where each tensor holds the stacked samples of an item.
            - I.e. output_data = {"item1": StackedSamplesOfItem1, "item2": StackedSamplesOfItem2, ...}
    """
    collate_timer = Timer("step/load_batch/collate_batch")
    # Restructure data to group samples by item key
    items = {key: [d[key] for d in data] for key in data[0].keys()}

    # Convert gripper history to tensor
    # Using statebase as its a common function to convert history to tensor.
    # Check if gripper history and gt gripper pred are in the items keys
    if POLICY_STATE_HISTORY_ITEM_NAME in items.keys():
        items[POLICY_STATE_HISTORY_ITEM_NAME] = state_tensor_from_history_list(
            items[POLICY_STATE_HISTORY_ITEM_NAME]
        )
    if GT_POLICY_STATE_PRED_ITEM_NAME in items.keys():
        items[GT_POLICY_STATE_PRED_ITEM_NAME] = state_tensor_from_history_list(
            items[GT_POLICY_STATE_PRED_ITEM_NAME]
        )

    # Stack the samples of each item
    stacked_samples_list = {}
    for item_name, samples_of_item in items.items():
        if isinstance(samples_of_item[0], torch.Tensor):
            stacked_samples = torch.stack(samples_of_item)
        elif isinstance(samples_of_item[0], dict):
            stacked_samples = batch_mesh_vertices(samples_of_item)
        else:
            raise NotImplementedError(type(samples_of_item[0]))

        stacked_samples_list[item_name] = stacked_samples

    collate_timer.stop()
    return stacked_samples_list


def check_batch_size(batch: Dict[str, torch.Tensor], batch_size: int):
    """Verifies that all tensors in the batch have the expected batch size.

    Checks tensors in dictionaries to ensure their first dimension matches
    the expected batch size.

    Args:
        batch (Dict[str, torch.Tensor]): Dictionary containing tensors
        batch_size (int): Expected size of the first dimension of all tensors

    Raises:
        AssertionError: If any tensor's batch dimension does not match the expected batch_size
    """
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert (
                batch_size == value.shape[0]
            ), f"Expected batch size {batch_size} but got {value.shape[0]} for key {key}"


def unpack_rgb(
    rgb_camera_item_names: List[str],
    batch: Dict[str, torch.Tensor],
    batch_size: int,
    image_size: Tuple[int, int],
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Unpacks a batch of RGB data and returns a dictionary of samples.

    Args:
        rgb_camera_item_names (List[str]): List of rgb camera item names to unpack
        batch_size (int): The batch size
        image_size (Tuple[int, int]): The size of the images
        device (str, optional): The device on which to perform the computation. Defaults to 'cuda'.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of samples.
    """
    samples = {}
    samples["rgbs"] = torch.stack(
        [batch[item_name] for item_name in rgb_camera_item_names], dim=1
    ).to(device)
    assert samples["rgbs"].shape == (
        batch_size,
        len(rgb_camera_item_names),
        3,
        image_size[0],
        image_size[1],
    ), f"Expected shape (batch_size, len(item_names), 3, image_size[0], image_size[1]) but got {samples['rgbs'].shape}"
    return samples


def structure_depth_camera_item_names(depth_camera_item_names: List[str]) -> List[Dict[str, str]]:
    """Converts flat list of depth camera-related item names into structured format by searching for keywords.

    Args:
        depth_camera_item_names: List of strings containing depth camera-related filenames

    Returns:
        List of dicts containing matched depth, pose, and intrinsics keys for each camera

    Raises:
        AssertionError: If the number of depth, pose, and intrinsics keys don't match
    """
    # Find all keys by type
    depth_item_names = [item_name for item_name in depth_camera_item_names if "depth" in item_name]
    pose_item_names = [item_name for item_name in depth_camera_item_names if "pose" in item_name]
    intrinsics_item_names = [
        item_name for item_name in depth_camera_item_names if "intrinsics" in item_name
    ]

    # Validate we have matching numbers of each type
    num_cameras = len(depth_item_names)
    assert (
        len(pose_item_names) == num_cameras
    ), f"Found {len(pose_item_names)} pose keys but {num_cameras} depth keys"
    assert (
        len(intrinsics_item_names) == num_cameras
    ), f"Found {len(intrinsics_item_names)} intrinsics keys but {num_cameras} depth keys"

    # Group by camera name prefix (e.g., 'pov' from 'pov_depth.png')
    structured_keys = []
    for depth_item_name in depth_item_names:
        prefix = depth_item_name.split("_")[0]  # Get camera prefix (e.g., 'pov')
        matching_pose = next(name for name in pose_item_names if name.startswith(prefix))
        matching_intrinsics = next(
            name for name in intrinsics_item_names if name.startswith(prefix)
        )

        structured_keys.append(
            {"depth": depth_item_name, "pose": matching_pose, "intrinsics": matching_intrinsics}
        )

    return structured_keys


def unpack_pcd(
    depth_camera_item_names: List[str],
    batch: Dict[str, torch.Tensor],
    batch_size: int,
    image_size: Tuple[int, int],
    rgbd_min_depth_threshold: float,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Unpacks a batch of PCD data and returns a dictionary of samples.

    Args:
        depth_camera_item_names (List[str]): List of depth camera item names to unpack
        batch_size (int): The batch size
        image_size (Tuple[int, int]): The size of the images
        device (str, optional): The device on which to perform the computation. Defaults to 'cuda'.
        rgbd_min_depth_threshold (float, optional): The depth threshold for valid PCD points.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of samples.
    """
    samples = {}
    samples["pcds"] = torch.stack(
        [
            get_camera_pointcloud(
                intrinsics=batch[camera_dict["intrinsics"]],
                depth=batch[camera_dict["depth"]],
                position=batch[camera_dict["pose"]][:, :3],
                orientation=batch[camera_dict["pose"]][:, 3:],
            )
            for camera_dict in structure_depth_camera_item_names(depth_camera_item_names)
        ],
        dim=1,
    ).to(device)

    samples["pcd_valid_mask"] = torch.stack(
        [
            batch[camera_dict["depth"]] > rgbd_min_depth_threshold
            for camera_dict in structure_depth_camera_item_names(depth_camera_item_names)
        ],
        dim=1,
    ).to(device)

    assert samples["pcds"].shape == (
        batch_size,
        len(structure_depth_camera_item_names(depth_camera_item_names)),
        3,
        image_size[0],
        image_size[1],
    )
    return samples


def unpack_policy_state(
    embodiment: EmbodimentBase,
    batch: Dict[str, torch.Tensor],
    batch_size: int,
    num_history: int,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Unpacks a batch of policy state history and returns a dictionary of samples."""
    samples = {}
    # Gripper history.
    policy_state_history = batch[POLICY_STATE_HISTORY_ITEM_NAME].to(device)
    # Dont check the last dimension, it varies depending on the embodiment.
    assert policy_state_history.shape[0] == batch_size
    # We split the policy state history based on the embodiment.
    samples["gripper_history"] = embodiment.policy_state_type.split_gripper_tensor(
        policy_state_history
    )

    # GT trajectory prediction.
    gt_policy_state_pred = batch[GT_POLICY_STATE_PRED_ITEM_NAME].to(device)
    assert gt_policy_state_pred.shape[0] == batch_size
    assert gt_policy_state_pred.shape[1] == gt_policy_state_pred.shape[1]
    # We split the gt policy state pred based on the embodiment (handle head yaw separately).
    if embodiment.embodiment_type == EmbodimentType.HUMANOID:
        samples["gt_head_yaw"] = embodiment.policy_state_type.split_head_yaw_tensor(
            gt_policy_state_pred
        )
    else:
        samples["gt_head_yaw"] = None
    samples["gt_gripper_pred"] = embodiment.policy_state_type.split_gripper_tensor(
        gt_policy_state_pred
    )

    # is_keypose flag.
    samples["is_keypose"] = batch[IS_KEYPOSE_ITEM_NAME].to(device)

    return samples


def unpack_mesh(
    batch: Dict[str, torch.Tensor], batch_size: int, device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Unpacks a batch of mesh data and returns a dictionary of samples."""
    samples = {}
    samples["vertex_features"] = (
        batch[NVBLOX_VERTEX_FEATURES_ITEM_NAME]["features"].to(torch.float32).to(device)
    )
    samples["vertices"] = (
        batch[NVBLOX_VERTEX_FEATURES_ITEM_NAME]["vertices"].to(torch.float32).to(device)
    )
    samples["vertices_valid_mask"] = batch[NVBLOX_VERTEX_FEATURES_ITEM_NAME][
        "vertices_valid_mask"
    ].to(device)
    channel_length = batch[NVBLOX_VERTEX_FEATURES_ITEM_NAME]["channel_length"]
    assert samples["vertex_features"].shape[0] == batch_size
    assert samples["vertex_features"].shape[2] == channel_length
    assert samples["vertices"].shape[0] == batch_size
    assert samples["vertices"].shape[2] == 3
    assert samples["vertices"].shape[1] == samples["vertex_features"].shape[1]
    assert samples["vertices_valid_mask"].shape[0] == batch_size
    assert samples["vertices_valid_mask"].shape[1] == samples["vertices"].shape[1]
    return samples


def unpack_batch(
    embodiment: EmbodimentBase,
    batch: torch.Tensor,
    batch_size: int,
    image_size: Tuple[int, int],
    num_history: int,
    data_type: DataType,
    feature_type: FeatureExtractorType,
    add_external_cam: bool,
    rgbd_min_depth_threshold: float = 0.0,
    device="cuda",
) -> Dict[str, torch.Tensor]:
    """
    Unpacks a batch of data and returns a dictionary of samples.

    Args:
        batch (torch.Tensor): The batch of data to unpack.
        batch_size (int): The batch size.
        image_size (Tuple[int, int]): The size of the images.
        num_history (int): The number of steps in the trajectory history.
        data_type (DataType): The type of data to unpack.
        feature_type (FeatureExtractorType): The type of feature to use.
        add_external_cam (bool): Whether to add the external cam data.
        rgbd_min_depth_threshold (float, optional): The depth threshold for valid PCD points. Defaults to 0.0.
        device (str, optional): The device on which to perform the computation. Defaults to 'cuda'.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of samples.
    """
    check_batch_size(batch, batch_size)

    samples = {
        "instr": None,
        "gripper_history": None,
        "gt_gripper_pred": None,
        "is_keypose": None,
        "rgbs": None,
        "pcds": None,
        "pcd_valid_mask": None,
        "depths": None,
        "segmentation_masks": None,
        "intrinsics": None,
        "camera_poses": None,
        "vertex_features": None,
        "vertices": None,
        "vertices_valid_mask": None,
    }

    if embodiment.embodiment_type == EmbodimentType.ARM:
        embodiment_specific_items = embodiment.get_camera_item_names_by_encoding_method(
            add_external_cam=add_external_cam
        )
    elif embodiment.embodiment_type == EmbodimentType.HUMANOID:
        embodiment_specific_items = embodiment.get_camera_item_names_by_encoding_method(
            add_external_cam=add_external_cam
        )
    else:
        raise ValueError(f"Unsupported embodiment type: {embodiment.embodiment_type}")

    if includes_policy_states(data_type):
        samples.update(unpack_policy_state(embodiment, batch, batch_size, num_history, device))

    if includes_rgb(data_type):
        samples.update(
            unpack_rgb(
                embodiment_specific_items["rgb"],
                batch,
                batch_size,
                image_size,
                device,
            )
        )

    if includes_pcd(data_type):
        samples.update(
            unpack_pcd(
                embodiment_specific_items["depth"],
                batch,
                batch_size,
                image_size,
                rgbd_min_depth_threshold,
                device,
            )
        )

    if includes_mesh(data_type):
        samples.update(unpack_mesh(batch, batch_size, device))

    return samples

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, WeightedRandomSampler

from mindmap.data_loading.data_types import (
    DataType,
    includes_depth_camera,
    includes_mesh,
    includes_rgb,
)
from mindmap.data_loading.dataset import SamplingWeightingType, get_dataloader
from mindmap.data_loading.item_names import (
    COMMON_RUNTIME_ITEMS,
    GT_POLICY_STATE_PRED_ITEM_NAME,
    MESH_ITEMS,
    NVBLOX_VERTEX_FEATURES_ITEM_NAME,
    POLICY_STATE_HISTORY_ITEM_NAME,
)
from mindmap.data_loading.sample_transformer import (
    DepthTransformer,
    GeometryAugmentor,
    GeometryNoiser,
    RgbTransformer,
    SampleTransformer,
    VertexSampler,
)
from mindmap.data_loading.vertex_sampling import VertexSamplingMethod
from mindmap.embodiments.embodiment_base import EmbodimentBase, EmbodimentType
from mindmap.keyposes.keypose_detection_mode import KeyposeDetectionMode
from mindmap.tasks.tasks import Tasks


def get_data_loader_by_data_type(
    embodiment: EmbodimentBase,
    dataset_path: str,
    demos: str,
    task: Tasks,
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
    apply_random_transforms: bool,
    apply_geometry_noise: bool,
    pos_noise_stddev_m: float,
    rot_noise_stddev_deg: float,
    add_external_cam: bool,
    num_vertices_to_sample: Optional[int] = None,
    vertex_sampling_method: Optional[VertexSamplingMethod] = None,
    random_translation_range_m: Tuple[List[float], List[float]] = None,
    random_rpy_range_deg: Tuple[List[float], List[float]] = None,
    seed: int = 0,
) -> Tuple[DataLoader, WeightedRandomSampler]:
    """
    Get a data loader for the given encoding method.

    Args:
        dataset_path (str): Path to the dataset.
        demos (str): The demos indices to load represented as a range string e.g. "0-5 7 9-11".
        task (Tasks): The task were loading data for.
        num_workers (int): Number of workers for the data loader.
        batch_size (int): Batch size for the data loader.
        use_keyposes (bool): Whether to use keyposes.
        data_type (DataType): The type of data to load.
        only_sample_keyposes (bool): Whether to only sample keyposes.
        extra_keyposes_around_grasp_events (List[int]): The number of extra keyposes to sample around grasp events.
        keypose_detection_mode (KeyposeDetectionMode): The keypose detection mode to use.
        include_failed_demos (bool): Whether to include failed demos.
        sampling_weighting_type (SamplingWeightingType): The sampling weighting type to use.
        gripper_encoding_mode (str): The gripper encoding mode to use.
        num_history (int): The number of history steps to use.
        prediction_horizon (int): The number of prediction steps to use.
        apply_random_transforms (bool): Whether to apply random transformations for data augmentation.
        apply_geometry_noise (bool): Whether to apply noise to 3d points and poses
        pos_noise_stddev_m: Standard deviation of geometry noise
        rot_noise_stddev_deg: Standard deviation of rotation noise
        random_translation_range_m: Range of random translation
        random_rpy_range_deg: Range of random rotation
        add_external_cam (bool): Whether to add the external cam data.
        num_vertices_to_sample: Number of vertices to sample
        vertex_sampling_method: Method to sample vertices
        seed: The seed to use for the data loader.
    """

    # Get emboidment specific items
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

    item_names = get_item_names_by_data_type(data_type, embodiment_specific_items)
    item_transforms = get_transforms_by_data_type(
        data_type=data_type,
        embodiment_specific_items=embodiment_specific_items,
        apply_random_transforms=apply_random_transforms,
        apply_geometry_noise=apply_geometry_noise,
        pos_noise_stddev_m=pos_noise_stddev_m,
        rot_noise_stddev_deg=rot_noise_stddev_deg,
        random_translation_range_m=random_translation_range_m,
        random_rpy_range_deg=random_rpy_range_deg,
        num_vertices_to_sample=num_vertices_to_sample,
        vertex_sampling_method=vertex_sampling_method,
        seed=seed,
    )

    data_loader, distributed_sampler = get_dataloader(
        dataset_path=dataset_path,
        embodiment=embodiment,
        demos=demos,
        task=task,
        item_names=item_names,
        transforms=item_transforms,
        num_workers=num_workers,
        batch_size=batch_size,
        use_keyposes=use_keyposes,
        only_sample_keyposes=only_sample_keyposes,
        extra_keyposes_around_grasp_events=extra_keyposes_around_grasp_events,
        keypose_detection_mode=keypose_detection_mode,
        include_failed_demos=include_failed_demos,
        sampling_weighting_type=sampling_weighting_type,
        data_type=data_type,
        gripper_encoding_mode=gripper_encoding_mode,
        num_history=num_history,
        prediction_horizon=prediction_horizon,
        seed=seed,
    )

    return data_loader, distributed_sampler


def get_data_loader_without_augmentations(
    embodiment: EmbodimentBase,
    dataset_path: str,
    demos: List[int],
    task: Tasks,
    num_workers: int,
    batch_size: int,
    use_keyposes: bool,
    data_type: DataType,
    extra_keyposes_around_grasp_events: List[int],
    keypose_detection_mode: KeyposeDetectionMode,
    gripper_encoding_mode: str,
    num_history: int,
    prediction_horizon: int,
    add_external_cam: bool,
    num_vertices_to_sample: int,
    sampling_weighting_type: SamplingWeightingType,
    vertex_sampling_method: VertexSamplingMethod,
    include_failed_demos: bool = False,
    seed: int = 0,
):
    """
    Get a data loader used for evaluation with the augmentations disabled.
    """
    return get_data_loader_by_data_type(
        embodiment=embodiment,
        dataset_path=dataset_path,
        demos=demos,
        task=task,
        num_workers=num_workers,
        batch_size=batch_size,
        use_keyposes=use_keyposes,
        data_type=data_type,
        only_sample_keyposes=False,
        extra_keyposes_around_grasp_events=extra_keyposes_around_grasp_events,
        keypose_detection_mode=keypose_detection_mode,
        include_failed_demos=include_failed_demos,
        gripper_encoding_mode=gripper_encoding_mode,
        num_history=num_history,
        prediction_horizon=prediction_horizon,
        num_vertices_to_sample=num_vertices_to_sample,
        vertex_sampling_method=vertex_sampling_method,
        add_external_cam=add_external_cam,
        sampling_weighting_type=sampling_weighting_type,
        apply_random_transforms=False,
        apply_geometry_noise=False,
        pos_noise_stddev_m=0.0,
        rot_noise_stddev_deg=0.0,
        random_translation_range_m=None,
        random_rpy_range_deg=None,
        seed=seed,
    )


def get_item_names_by_data_type(data_type: DataType, embodiment_specific_items: Dict) -> List[str]:
    item_names = []
    item_names.extend(COMMON_RUNTIME_ITEMS)

    # Camera items
    if includes_rgb(data_type):
        item_names.extend(embodiment_specific_items["rgb"])
    if includes_depth_camera(data_type):
        item_names.extend(embodiment_specific_items["depth"])

    # Reconstruction items
    if includes_mesh(data_type):
        item_names.extend(MESH_ITEMS)

    return item_names


# Disable yapf to not break the type hints into multiple lines.
# yapf: disable
def get_transforms_by_data_type(
        data_type: DataType,
        embodiment_specific_items: Dict,
        apply_random_transforms: bool,
        apply_geometry_noise: bool,
        pos_noise_stddev_m: float,
        rot_noise_stddev_deg: float,
        random_translation_range_m: Tuple[List[float], List[float]],
        random_rpy_range_deg: Tuple[List[float], List[float]],
        num_vertices_to_sample: Optional[int] = None,
        vertex_sampling_method: Optional[VertexSamplingMethod] = None,
        seed: int = None) -> Dict[str, List[SampleTransformer]]:
    # yapf: enable
    """
    Creates a dictionary of transforms for the given encoding method.

    Args:
        data_type (DataType, optional): The type of data to transform.
        apply_random_transforms (bool): Whether to apply random transformations for data augmentation.
        apply_geometry_noise (bool): Whether to apply noise to 3d points and poses
        pos_noise_stddev_m: Standard deviation of geometry noise
        rot_noise_stddev_deg: Standard deviation of rotation noise
        random_translation_range_m: Range of random translation
        random_rpy_range_deg: Range of random rotation
        num_vertices_to_sample: Number of vertices to sample
        vertex_sampling_method: Method to sample vertices
    """
    assert (
        random_translation_range_m is not None and random_rpy_range_deg is not None
    ) or not apply_random_transforms, 'random_translation_range_m and random_rpy_range_deg must be set if apply_random_transforms is True'

    transforms = defaultdict(list)
    if apply_random_transforms:
        # Data augmentation transforms.
        # All items must share the same transformer object in order to ensure that the same
        # transform is applied to all of them
        augmentor = GeometryAugmentor(random_translation_range_m, random_rpy_range_deg)
        transforms[POLICY_STATE_HISTORY_ITEM_NAME].append(augmentor)
        transforms[GT_POLICY_STATE_PRED_ITEM_NAME].append(augmentor)
        if data_type == DataType.MESH:
            transforms[NVBLOX_VERTEX_FEATURES_ITEM_NAME].append(augmentor)
        else:
            raise NotImplementedError(f"Applying random transforms to unsupported input data type: {data_type}")

    if apply_geometry_noise:
        # Transforms that add noise. We apply only to history data, not to GT
        noiser = GeometryNoiser(pos_noise_stddev_m, rot_noise_stddev_deg)
        transforms[POLICY_STATE_HISTORY_ITEM_NAME].append(noiser)
        if includes_mesh(data_type):
            transforms[NVBLOX_VERTEX_FEATURES_ITEM_NAME].append(noiser)
        else:
            raise NotImplementedError(f"Applying geometry noise to unsupported input data type: {data_type}")

    if includes_rgb(data_type):
        for rgb_item in embodiment_specific_items['rgb']:
            transforms[rgb_item].append(RgbTransformer())

    if includes_depth_camera(data_type):
        for depth_item in embodiment_specific_items['depth']:
            # Only do it for the depth image, and not the intrinsics or pointcloud
            if "png" in depth_item:
                transforms[depth_item].append(DepthTransformer())

    if includes_mesh(data_type):
        # Subsample mesh vertices to correct size.
        transforms[NVBLOX_VERTEX_FEATURES_ITEM_NAME].append(
            VertexSampler(
                desired_num_vertices=num_vertices_to_sample,
                method=vertex_sampling_method,
                seed=seed))
    return transforms

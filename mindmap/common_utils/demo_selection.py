# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import List, Optional, Tuple

DEMO_PATH_NUM_DIGITS = 5


def get_num_demos(demos: str) -> int:
    """Get the number of demos from a range string.

    Args:
        demos: A string containing demo indices in range format, e.g. "0-5 7 9-11"
            would represent demos [0,1,2,3,4,5,7,9,10,11]

    Returns:
        The total number of demos specified in the range string
    """
    return len(get_indices_from_range_str(demos))


def get_episode_names(demos: str) -> List[str]:
    """Get a list of episode names from a range string.

    Args:
        demos: A string containing demo indices in range format, e.g. "0-5 7 9-11"
            would represent demos [0,1,2,3,4,5,7,9,10,11]

    Returns:
        A list of episode names in the format "demo_X" where X is the demo index
    """
    demo_indices = get_indices_from_range_str(demos)
    return list(map(lambda demo_index: get_demo_name(demo_index), demo_indices))


def get_demo_paths(dataset_path: str, demos: str) -> List[str]:
    """Get a list of demo paths from a dataset path and range string.

    Args:
        dataset_path: Path to the dataset directory
        demos: A string containing demo indices in range format, e.g. "0-5 7 9-11"
            would represent demos [0,1,2,3,4,5,7,9,10,11]

    Returns:
        A sorted list of full paths to each demo directory specified in the range string
    """
    demo_indices = get_indices_from_range_str(demos)
    return sorted(
        list(map(lambda demo_index: get_demo_path(dataset_path, demo_index), demo_indices))
    )


def get_demo_path(dataset_path: str, demo_index: int) -> str:
    """Get the full path to a demo directory.

    Args:
        dataset_path: Path to the dataset directory
        demo_index: Index of the demo

    Returns:
        Full path to the demo directory with zero-padded index (i.e. <dataset_path>/demo_00000)
    """
    return f"{dataset_path}/{get_demo_name(demo_index, DEMO_PATH_NUM_DIGITS)}"


def get_demo_name(demo_index: int, num_digits: int = None) -> str:
    """Get the name of a demo directory.

    Args:
        demo_index: Index of the demo
        num_digits: Optional number of digits to zero-pad the demo index. If None,
            no zero padding is applied.

    Returns:
        Name of the demo directory (i.e. 'demo_00000' if num_digits=5, or 'demo_0' if num_digits=None)
    """
    if num_digits is None:
        return f"demo_{demo_index}"
    else:
        return f"demo_{demo_index:0{num_digits}d}"


def get_indices_from_range_str(multi_range_str: str) -> List[int]:
    """Get a list of indices from a range string.

    Args:
        multi_range_str: A string containing indices in range format, e.g. "0-5 7 9-11"
            would represent indices [0,1,2,3,4,5,7,9,10,11]. Multiple ranges can be
            specified by separating with spaces.

    Returns:
        A sorted list of integers representing the indices specified in the range string.
    """
    indices = []
    for range_str in multi_range_str.split(" "):
        if "-" in range_str:
            start, end = map(int, range_str.split("-"))
            assert start <= end
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(range_str))
    return sorted(indices)


def min_max_from_range(range_str: str) -> Tuple[int, int]:
    """Get the min and max from a range string."""
    indices = get_indices_from_range_str(range_str)
    min_idx = min(indices)
    max_idx = max(indices)
    assert min_idx <= max_idx
    return min_idx, max_idx


def get_demo_super_range(demos_str: str, demos_valset_str: Optional[str] = None) -> str:
    """Combine the demos from the training and valset ranges into a single range."""
    # Demos min and max.
    demos_min, demos_max = min_max_from_range(demos_str)
    # Valset min and max.
    if demos_valset_str is not None:
        demos_valset_min, demos_valset_max = min_max_from_range(demos_valset_str)
        demos_min, demos_max = min(demos_min, demos_valset_min), max(demos_max, demos_valset_max)
    assert demos_min <= demos_max
    # Build the output range string.
    if demos_min == demos_max:
        output_str = str(demos_min)
    else:
        output_str = f"{demos_min}-{demos_max}"
    return output_str

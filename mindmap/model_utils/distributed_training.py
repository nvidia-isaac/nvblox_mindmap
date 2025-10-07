# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import pickle

import torch
import torch.distributed as dist


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)

    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """
    Returns the number of processes in the default group.
    This function checks if the distributed package is available and initialized.
    If it is not available or not initialized, it returns 1.
    Otherwise, it calls the `get_world_size` function from the `dist` module
    and returns the result.

    Returns:
        int: The number of processes in the default group.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """
    Returns the rank of the current process in the default group.
    This function checks if the distributed package is available and initialized.
    If it is not available or not initialized, it returns 0.
    Otherwise, it calls the `get_rank` function from the `dist` module
    and returns the result.
    Returns:
        int: The rank of the current process in the default group.
    """
    if not is_dist_avail_and_initialized():
        return 0
    else:
        return dist.get_rank()


def print_dist(*args, **kwargs):
    """
    Calling print function if the rank of the current process is 0.
    Can be used to prevent printing from multiple processes.
    """
    if get_rank() == 0:
        print(*args, **kwargs)

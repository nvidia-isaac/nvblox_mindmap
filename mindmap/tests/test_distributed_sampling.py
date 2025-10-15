# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import os

from catalyst.data.sampler import DistributedSamplerWrapper
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import WeightedRandomSampler


def get_weighted_random_sampler(weights, num_samples, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    weighted_sampler = WeightedRandomSampler(
        weights, num_samples, generator=generator, replacement=True
    )
    return weighted_sampler


def init(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def destroy():
    dist.destroy_process_group()


def run_mp_sampling(rank, world_size, epoch_idx, seed, weights, shared_list):
    init(rank, world_size)

    # Sample with distributed weighted random sampler
    weighted_sampler = get_weighted_random_sampler(weights, len(weights), seed)
    distributed_weighted_sampler = DistributedSamplerWrapper(sampler=weighted_sampler, shuffle=True)
    distributed_weighted_sampler.set_epoch(epoch_idx)

    # Every process adds its samples into a shared list
    shared_list.extend(list(distributed_weighted_sampler))

    destroy()


def test_distributed_sampling():
    """
    Test the distributed sampling functionality (imported from catalyst) in combination with weighted random sampling.

    This function tests the distributed sampling functionality by comparing the indices sampled
    using a weighted random sampler in a single process with the indices sampled using a
    distributed weighted random sampler in multiple processes.
    """
    seed = 10
    num_samples = 100
    world_size = 4  # number of processes
    epoch_idx = 0  # used for seeding the shuffling
    weights = torch.rand(num_samples)

    print("Sample with weighted random sampler (single-process).")
    sampled_indices = torch.tensor(
        sorted(list(get_weighted_random_sampler(weights, num_samples, seed)))
    )

    print("Sample with distributed weighted random sampler (multi-process).")
    manager = mp.Manager()
    mp_combined_sampled_indices = manager.list()
    mp.spawn(
        run_mp_sampling,
        args=(world_size, epoch_idx, seed, weights, mp_combined_sampled_indices),
        nprocs=world_size,
        join=True,
    )
    mp_combined_sampled_indices = torch.tensor(sorted(mp_combined_sampled_indices))

    print(
        "Check that we sampled the same indices using distributed sampling as with single process."
    )
    assert torch.eq(mp_combined_sampled_indices, sampled_indices).all()

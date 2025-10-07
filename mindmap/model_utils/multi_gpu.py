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

import torch
import torch.distributed as dist


class MultiProcessGroup:
    """
    Context manager for initializing and destroying a torch distributed process group.
    """

    def __init__(self, backend="nccl", init_method="env://"):
        self.backend = backend
        self.init_method = init_method
        self.initialized = False
        self.local_rank = int(os.environ["LOCAL_RANK"])

    def __enter__(self):
        if not dist.is_initialized():
            print(f"[rank: {self.local_rank}] Initializing process group")
            dist.init_process_group(backend=self.backend, init_method=self.init_method)
            self.initialized = True
            torch.cuda.set_device(self.local_rank)
            print(f"[rank: {self.local_rank}] Initialized process group")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.initialized and dist.is_initialized():
            print(f"[rank: {self.local_rank}] Destroying process group")
            dist.destroy_process_group()
            self.initialized = False

    def get_local_rank(self):
        return self.local_rank

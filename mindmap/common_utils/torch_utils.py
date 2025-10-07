# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import torch


class AllowMatMulTf32:
    """Context manager for enabling  matrix multiplication using TensorFloat32.
    This will improve computation speed on supported hardware to the cost of reduced precision."""

    def __enter__(self):
        self.old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True

    def __exit__(self, exc_type, exc_value, traceback):
        torch.backends.cuda.matmul.allow_tf32 = self.old_allow_tf32

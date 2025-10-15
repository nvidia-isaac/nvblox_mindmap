# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
def print_num_trainable_params(model, level=0, max_level=2):
    """Recursive function to count trainable parameters for model + submodels

    Args:
        model        Model to print parameters for
        level        Recursion parameter. Current level in the hierarchy
        max_level    Maximum level to vist.

    Return:
        Total number of parameters
    """

    total_params = None
    if level == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of parameters: {total_params}")

    for name, submodule in model.named_children():
        num_params = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
        if num_params > 0:
            print(
                " " * 2 * level + f".{name}: {num_params} trainable parameters (including children)"
            )
        # Recursively check submodules
        if level < max_level:
            print_num_trainable_params(submodule, level + 1, max_level)

    return total_params

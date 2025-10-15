# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# NVblox items
NVBLOX_VERTEX_FEATURES_ITEM_NAME = "nvblox_vertex_features.zst"

# Runtime items
POLICY_STATE_HISTORY_ITEM_NAME = "runtime_policy_state_history"
GT_POLICY_STATE_PRED_ITEM_NAME = "runtime_gt_policy_state_pred"
IS_KEYPOSE_ITEM_NAME = "runtime_is_keypose"


# The common runtime entries (gripper history, prediction, and keypose info) are added to all methods.
COMMON_RUNTIME_ITEMS = [
    POLICY_STATE_HISTORY_ITEM_NAME,
    GT_POLICY_STATE_PRED_ITEM_NAME,
    IS_KEYPOSE_ITEM_NAME,
]


MESH_ITEMS = [
    NVBLOX_VERTEX_FEATURES_ITEM_NAME,
]

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from enum import Enum


class ClosedLoopMode(Enum):
    CLOSED_LOOP_WAIT = "closed_loop_wait"  # run closed loop inference, only run inference once last predicted goal is reached.
    EXECUTE_GT_GOALS = "execute_gt_goals"  # execute the GT predictions as commands to the arm.
    DUMMY = "dummy"  # execute some pre-defined dummy goals.

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from mindmap.embodiments.embodiment_base import EmbodimentType
from mindmap.tasks.tasks import Tasks

TASK_TO_EMBODIMENT_TYPE = {
    Tasks.CUBE_STACKING: EmbodimentType.ARM,
    Tasks.MUG_IN_DRAWER: EmbodimentType.ARM,
    Tasks.DRILL_IN_BOX: EmbodimentType.HUMANOID,
    Tasks.STICK_IN_BIN: EmbodimentType.HUMANOID,
}


def get_embodiment_type_from_task(task: Tasks | str) -> EmbodimentType:
    """Get the embodiment type from the task."""
    if isinstance(task, str):
        task = Tasks(task)
    return TASK_TO_EMBODIMENT_TYPE[task]

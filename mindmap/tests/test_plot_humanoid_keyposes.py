# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import pathlib

from mindmap.scripts.plot_humanoid_keyposes import main
from mindmap.tests.utils.constants import TestDataLocations


def test_plot_humanoid_keyposes():
    # Tests that the script can interpret the generated demo data
    humanoid_generated_demo_path = (
        pathlib.Path(TestDataLocations.generated_data_dir) / "drill_in_box" / "demo_00000"
    )
    main(data_path=humanoid_generated_demo_path, plot=False)

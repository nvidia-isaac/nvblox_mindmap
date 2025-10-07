# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import glob
import os
import pathlib

from mindmap.scripts.tar_demos import tar_demos
from mindmap.scripts.untar_demos import untar_demos


def test_tar_untar_demos(tmp_path):
    # Create some folders to work in.
    dummy_data_dir = tmp_path / "dummy_data"
    tarred_data_dir = tmp_path / "tarred"
    untarred_data_dir = tmp_path / "untarred"
    os.makedirs(dummy_data_dir, exist_ok=True)
    os.makedirs(tarred_data_dir, exist_ok=True)
    os.makedirs(untarred_data_dir, exist_ok=True)

    # Create dummy demo files
    demo_names = ["demo_00000", "demo_00001"]
    test_file_names = ["test_1", "test_2", "test_3"]
    for demo_name in demo_names:
        demo_dir = dummy_data_dir / demo_name
        os.makedirs(demo_dir, exist_ok=True)
        for file_name in test_file_names:
            file_path = demo_dir / file_name
            with open(file_path, "w") as f:
                f.write(f"{file_name}")

    # Tar and then untar the demos.
    tar_demos(demos_dir=dummy_data_dir, output_dir=tarred_data_dir)
    untar_demos(demos_dir=tarred_data_dir, output_dir=untarred_data_dir)

    # Check we have all our files and their contents are correct
    for demo_dir in glob.glob(str(untarred_data_dir / "demo_*")):
        demo_dir = pathlib.Path(demo_dir)
        assert demo_dir.name in demo_names
        for file in glob.glob(str(demo_dir / "*")):
            file_path = pathlib.Path(file)
            assert file_path.name in test_file_names
            with open(file, "r") as f:
                content = f.read()
                assert content == file_path.name

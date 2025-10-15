# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from setuptools import find_packages, setup

MINDMAP_VERSION_NUMBER = "1.0.0"


if __name__ == "__main__":
    setup(
        name="mindmap",
        version=MINDMAP_VERSION_NUMBER,
        description="A package containing scripts and tools for data generation,"
        " training and running of an imitation learning model with 3d perception input.",
        author="Alex Millane, Remo Steiner, Clemens Volk, David Tingdahl, Xinjie Yao, Peter Du, Vikram Ramasamy, Shiwei Sheng",
        author_email="nvblox-dev@exchange.nvidia.com",
        packages=find_packages(include=["mindmap", "mindmap.*", "mindmap_osmo", "mindmap_osmo.*"]),
        include_package_data=False,
    )

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import pytest

from mindmap.common_utils.demo_selection import get_demo_super_range


def test_get_demo_super_range():
    demos = "0-89"
    demos_valset = "90-99"
    output_range = get_demo_super_range(demos, demos_valset)
    assert output_range == "0-99"

    demos = "0-89"
    demos_valset = "99"
    output_range = get_demo_super_range(demos, demos_valset)
    assert output_range == "0-99"

    demos = "0"
    demos_valset = "99"
    output_range = get_demo_super_range(demos, demos_valset)
    assert output_range == "0-99"

    demos = "0"
    demos_valset = "0"
    output_range = get_demo_super_range(demos, demos_valset)
    assert output_range == "0"

    demos = "99"
    demos_valset = "99"
    output_range = get_demo_super_range(demos, demos_valset)
    assert output_range == "99"

    demos = "99"
    demos_valset = "0"
    output_range = get_demo_super_range(demos, demos_valset)
    assert output_range == "0-99"

    demos = "0-89"
    demos_valset = None
    output_range = get_demo_super_range(demos, demos_valset)
    assert output_range == "0-89"

    with pytest.raises(AssertionError):
        demos = "1-0"
        demos_valset = None
        output_range = get_demo_super_range(demos, demos_valset)
        assert output_range == "0-89"

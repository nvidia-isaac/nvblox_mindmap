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

# This file is used to configure the pytest environment.


def pytest_addoption(parser):
    """Add test options from cmdline"""
    parser.addoption(
        "--generate_baseline",
        action="store_true",
        help="Generate new baseline for regression tests.",
    )


@pytest.fixture
def generate_baseline_arg(request):
    """Tests inheriting this fixture can be passed the --generate_baseline flag from the pytest command line."""
    return request.config.getoption("--generate_baseline")

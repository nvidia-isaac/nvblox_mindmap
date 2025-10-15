# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import random
import subprocess


def get_shmem_usage_mb():
    """Return system-wide shared memory usage in bytes. This gives an indication on how much data
    we're storing in the data processing queues"""

    # Run command to get shared memory usage from /proc/meminfo
    result = subprocess.run(
        ["cat", "/proc/meminfo"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    shared_memory_kb = 0
    # Filter the line containing "Shmem" using string search
    for line in result.stdout.splitlines():
        if line.startswith("Shmem:"):
            shared_memory_kb = int(line.split()[1])

    return shared_memory_kb >> 10  # To mb


def get_random_port_in_unassigned_range() -> int:
    """Return a random port number that is not used by any known protocol."""
    # TODO(dtingdahl) Port is hardcoded to the default torchrun port for now due to suspected issues when running on OSMO. Investigate why this is the case.
    return 29400
    # return random.randint(49152, 65535)

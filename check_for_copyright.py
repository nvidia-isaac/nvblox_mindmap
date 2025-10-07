#!/usr/bin/env python
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import re
import sys
from typing import List

pattern_nvidia_copyright = r"""# Copyright \(c\) 2025 NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto\. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited\.
"""

pattern = re.compile(pattern_nvidia_copyright, re.MULTILINE | re.DOTALL)


def check_copyright(files: List[str]) -> int:
    files_missing_copyright = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            if not pattern.search(content):
                files_missing_copyright.append(file)

    if files_missing_copyright:
        for file in files_missing_copyright:
            print(f"The following file is missing a copyright: {file}")
        return 1
    return 0


def main() -> None:
    files = sys.argv[1:]
    sys.exit(check_copyright(files))


if __name__ == "__main__":
    main()

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


def init_script_snippet() -> str:
    """Returns bash script snippet for initializing the workflow environment.
    Sets up virtual env, system monitoring, filebrowser, and headless mode."""
    return """
set -euxo pipefail

# Activate the virtual environment
. /opt/venv/bin/activate

# Display info on the system
/print_system_status.sh

# Launch a filebrowser background process to access data and checkpoints.
# The on_exit() signal handler will ensure that filebrowser is terminated when the script exits.
# Failure to do so will lead to stalled jobs in OSMO.
on_exit() {
  echo "Running on_exit()"
  kill $(pidof filebrowser)  || true
}
trap on_exit EXIT
filebrowser config init
filebrowser users add mindmap_user mindmap_password
filebrowser -r / --username mindmap_user &

# If we run sim, do it headless.
export HEADLESS=1
"""


def untar_demos_snippet(ncpus: int, demos_dir: str) -> str:
    """Returns script snippet for extracting demo tar files in parallel.

    Args:
        ncpus: Number of CPU processes to use for extraction
        demos_dir: Directory containing demo tar files
    """
    return f"""
# Untar the demos
python scripts/untar_demos.py \\
    --demos_dir {demos_dir} \\
    --num_processes {ncpus} \\
    --remove_tar
"""


def tar_demos_snippet(ncpus: int, demos_dir: str = "{{output}}") -> str:
    """Returns script snippet for compressing demos into tar files in parallel.

    Args:
        ncpus: Number of CPU processes to use for tarring
    """
    snippet = f"""
python scripts/tar_demos.py \\
    --demos_dir {demos_dir} \\
    --num_processes {ncpus} \\
"""
    return snippet

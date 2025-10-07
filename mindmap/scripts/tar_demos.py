#!/usr/bin/env python
#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import argparse
from functools import partial
import glob
import multiprocessing
import os
import pathlib
import shutil
import tarfile
from typing import List, Optional


def create_tar(
    input_dir: str,
    output_dir: Optional[str] = None,
    remove_dir=False,
):
    """Converts a directory into a tar file.

    Args:
        input_dir (str): Path to the directory to be tarred.
        output_dir (Optional[str], optional): The directory to put the resulting tar file in.
            Defaults to None. Default is to put it in the parent directory of the input dir.
        remove_dir (bool, optional): Whether to remove the input dir. Defaults to False.
    """
    input_dir = pathlib.Path(input_dir)
    if not output_dir:
        output_dir = str(input_dir.parent)
    tar_path = os.path.join(output_dir, f"{input_dir.name}.tar")

    # Take the number of threads on the machine.
    print(f"Tarring demo at: {input_dir}")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(input_dir, arcname=input_dir.name)

    if remove_dir:
        shutil.rmtree(input_dir)


def tar_demos(
    demos_dir: str,
    output_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
):
    """Converts a folder of demos into a folder of tar files.

    Args:
        demos_dir (str): The folder containing the demo folders of form demo_XXXXX.
        output_dir (Optional[str], optional): The optional folder to put the tarred demos in.
            Defaults to None. Default behaviour is to put them in the input folder. In this
            case the input files will be deleted.
        num_workers (Optional[int], optional): Number of processes to use. Defaults to None.
            Default behaviour is to use one worker per CPU core.
    """
    # Grab the demo tar files
    demo_dirs = [d for d in glob.glob(os.path.join(demos_dir, "demo_*")) if os.path.isdir(d)]
    print(f"Found {len(demo_dirs)} demos to tar.")

    # Number of processes
    if num_workers == None:
        num_workers = min(multiprocessing.cpu_count(), len(demo_dirs))
    print(f"Using {num_workers} workers")

    with multiprocessing.Pool(num_workers) as pool:
        remove_dir = True if output_dir is None else False
        f = partial(
            create_tar,
            output_dir=output_dir,
            remove_dir=remove_dir,
        )
        pool.map(f, demo_dirs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tar demonstrations for Isaac Lab environments.")
    parser.add_argument(
        "--demos_dir", type=str, required=True, help="The directory containing extracted demos."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The directory to save the tar files. Default is the same directory."
        "In this default mode, the input (untarred) data is deleted.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="The number of processes to launch to untar.",
    )
    args = parser.parse_args()

    tar_demos(args.demos_dir, args.output_dir, args.num_processes)

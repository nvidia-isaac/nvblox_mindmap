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
import tarfile
from typing import Optional


def extract_tar(input_tar_path: str, output_dir: Optional[str] = None, remove_tar=True):
    """Converts a directory into a tar file.

    Args:
        input_dir (str): Path to the directory to be untarred.
        output_dir (Optional[str], optional): The directory to put the resulting data folder in.
            Defaults to None. Default is to put it in the parent directory of the input tar.
        remove_tar (bool, optional): Whether to remove the input tar. Defaults to True.
    """
    print(f"Untarring demo at: {input_tar_path}")
    if not output_dir:
        output_dir = str(pathlib.Path(input_tar_path).parent)
    with tarfile.open(input_tar_path, "r") as tar:
        tar.extractall(output_dir)
    if remove_tar:
        os.remove(input_tar_path)


def untar_demos(
    demos_dir: str,
    output_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
    remove_tar=False,
):
    """Converts a folder of demo tars into a folder of demo dirs.

    Args:
        demos_dir (str): The folder containing the demo folders of form demo_XXXXX.tar.
        output_dir (Optional[str], optional): The optional folder to put the untarred demos in.
            Defaults to None. Default behaviour is to put them in the input folder. In this
            case the input tar files will be deleted.
        num_workers (Optional[int], optional): Number of processes to use. Defaults to None.
            Default behaviour is to use one worker per CPU core.
        remove_tar (bool, optional): Whether to remove the input tar files. Defaults to False.
    """
    # Grab the demo tar files
    tar_files_list = glob.glob(os.path.join(demos_dir, "demo_*.tar"))
    print(f"Found {len(tar_files_list)} demos.")

    if num_workers == None:
        # Take the number of threads on the machine.
        num_workers = min(multiprocessing.cpu_count(), len(tar_files_list))
    print(f"Using {num_workers} workers")

    with multiprocessing.Pool(num_workers) as pool:
        f = partial(extract_tar, output_dir=output_dir, remove_tar=remove_tar)
        pool.map(f, tar_files_list)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description="Collect demonstrations for Isaac Lab environments."
    )
    parser.add_argument(
        "--demos_dir", type=str, required=True, help="The directory containing tarred demos."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The directory in which to dump the extracted demos. Default is to dump in the same directory.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="The number of processes to launch to untar.",
    )
    parser.add_argument(
        "--remove_tar", action="store_true", help="Whether to remove the input tar files."
    )
    args = parser.parse_args()
    # Untar
    untar_demos(args.demos_dir, args.output_dir, args.num_processes, args.remove_tar)

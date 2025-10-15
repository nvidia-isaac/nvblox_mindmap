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

import h5py


def print_level(level, indent=0):
    if isinstance(level, h5py.Dataset):
        return
    for key in level.keys():
        if isinstance(level[key], h5py.Dataset):
            shape = level[key].shape
            dtype = level[key].dtype
        else:
            shape = None
            dtype = None
        print(f'{" " * indent}{key}: type {type(level[key])}, shape {shape}, dtype {dtype}')
        print_level(level[key], indent + 2)


def print_demo_hdf5_structure(hdf5_path: str):
    with h5py.File(hdf5_path, "r") as f:
        print_level(f["data"]["demo_0"], 0)


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Extract a single demo from an HDF5 file")
    parser.add_argument("--input", type=str, required=True, help="Path to input HDF5 file")
    parser.add_argument("--output", type=str, required=True, help="Path to output HDF5 file")
    return parser


def main(input_hdf5_path: str, output_hdf5_path: str):
    print("input hdf5 file")
    print("-------------")
    print_demo_hdf5_structure(input_hdf5_path)

    with h5py.File(output_hdf5_path, "w") as f_out:
        with h5py.File(input_hdf5_path, "r") as f_in:
            # Get the data
            data_in = f_in["data"]
            data_out = f_out.create_group("data")
            f_in.copy("data/demo_0", f_out, "data/demo_0")
            # Copy over the env_args
            data_out.attrs["env_args"] = data_in.attrs["env_args"]

    print("\n\n\noutput hdf5 file")
    print("-------------")
    print_demo_hdf5_structure(output_hdf5_path)


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args.input, args.output)

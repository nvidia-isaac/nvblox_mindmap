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
import ast

import h5py


def merge_hdf5_files(inputs: str, output: str):
    with h5py.File(output, "w") as f_out:
        data_out = f_out.create_group("data")

        # Open all input files
        input_files = []

        for input in inputs:
            f_in = h5py.File(input, "r")
            input_files.append(f_in)

        try:
            # Set env_args from first input file
            data_in = input_files[0]["data"]

            env_args_str = data_in.attrs["env_args"]
            env_args_dict = ast.literal_eval(env_args_str)
            print(f"Env name: {env_args_dict['env_name']}")
            data_out.attrs["env_args"] = data_in.attrs["env_args"]

            # Assume all input files have the same number of demos
            num_demos = len(input_files[0]["data"].keys())

            # Check that all input files have the same number of demos
            for input_path, f_in in zip(inputs, input_files):
                file_num_demos = len(f_in["data"].keys())
                if file_num_demos != num_demos:
                    raise ValueError(
                        f"{input_path} has {file_num_demos} demos, but expected {num_demos} demos (same as first file)"
                    )

            # For each demo index, copy from each input file in order
            curr_output_demo_idx = 0
            for demo_idx in range(num_demos):
                for input_path, f_in in zip(inputs, input_files):
                    input_demo_name = f"demo_{demo_idx}"
                    output_demo_name = f"demo_{curr_output_demo_idx}"
                    print(
                        f"Copying {input_demo_name} of {input_path} to {output_demo_name} of {output}"
                    )
                    f_in.copy(f"data/{input_demo_name}", f_out, f"data/{output_demo_name}")
                    curr_output_demo_idx += 1

        finally:
            # Close all input files
            for f_in in input_files:
                f_in.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge hdf5 files into one")
    parser.add_argument(
        "--inputs", type=str, nargs="+", required=True, help="Path to the hdf5 files"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Where to save the merged hdf5 file"
    )
    args = parser.parse_args()

    merge_hdf5_files(args.inputs, args.output)

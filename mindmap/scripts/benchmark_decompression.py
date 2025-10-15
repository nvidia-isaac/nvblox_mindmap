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
import gzip

# Benchmark of decompression speed for various compression algorithms
#
# run pip install python-snappy zstandard lz4  brotli
import os
import shutil
import tempfile

import brotli
import lz4.frame
from nvblox_torch.timer import Timer, print_timers
import snappy
import zstandard


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark compression algorithms.")
    parser.add_argument("target", type=str, help="Path to the file to use for benchmarking.")

    # Parse arguments and call the main function
    return parser.parse_args()


NUM_ATTEMPTS = 1


def print_size(compressed_path, input_size, timer_name):
    size = os.path.getsize(compressed_path)
    ratio = input_size / size
    size_mb = size / 2**20
    print(f"{timer_name} compressed size: {size_mb:>.2f}, ratio: {ratio:>.2f}")


def benchmark_gzip(target, level, timer_name):
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(NUM_ATTEMPTS):
            compressed_path = os.path.join(tmp_dir, f"file.{i:>05}.gz")
            with open(target, "rb") as infile, gzip.open(
                compressed_path, "wb", compresslevel=level
            ) as outfile:
                shutil.copyfileobj(infile, outfile)

            with Timer(timer_name) as _:
                with gzip.open(compressed_path, "rb") as infile:
                    dummy = infile.read()

        print_size(compressed_path, os.path.getsize(target), timer_name)


def benchmark_lz4(target, level, timer_name):
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(NUM_ATTEMPTS):
            compressed_path = os.path.join(tmp_dir, f"file.{i:>05}.lz4")
            with open(target, "rb") as infile, open(compressed_path, "wb") as outfile:
                input_bytes = infile.read()
                outfile.write(lz4.frame.compress(input_bytes, compression_level=level))

            with Timer(timer_name) as _:
                with open(compressed_path, "rb") as infile:
                    decompressed_bytes = lz4.frame.decompress(infile.read())

        assert input_bytes == decompressed_bytes
        print_size(compressed_path, len(input_bytes), timer_name)


def benchmark_zstd(target, level, timer_name):
    cctx = zstandard.ZstdCompressor(level=level)
    dctx = zstandard.ZstdDecompressor()

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(NUM_ATTEMPTS):
            compressed_path = os.path.join(tmp_dir, f"file.{i:>05}.zst")
            with open(target, "rb") as infile, open(compressed_path, "wb") as outfile:
                input_bytes = infile.read()
                outfile.write(cctx.compress(input_bytes))

            with Timer(timer_name) as _:
                with open(compressed_path, "rb") as infile:
                    decompressed_bytes = dctx.decompress(infile.read())

        assert input_bytes == decompressed_bytes
        print_size(compressed_path, len(input_bytes), timer_name)


def benchmark_snappy(target, timer_name):
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(NUM_ATTEMPTS):
            compressed_path = os.path.join(tmp_dir, f"file.{i:>05}.zst")
            with open(target, "rb") as infile, open(compressed_path, "wb") as outfile:
                input_bytes = infile.read()
                outfile.write(snappy.compress(input_bytes))

            with Timer(timer_name) as _:
                with open(compressed_path, "rb") as infile:
                    decompressed_bytes = snappy.decompress(infile.read())

        assert input_bytes == decompressed_bytes
        print_size(compressed_path, len(input_bytes), timer_name)


def benchmark_brotli(target, level, timer_name):
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(NUM_ATTEMPTS):
            compressed_path = os.path.join(tmp_dir, f"file.{i:>05}.zst")
            with open(target, "rb") as infile, open(compressed_path, "wb") as outfile:
                input_bytes = infile.read()
                outfile.write(brotli.compress(input_bytes, quality=level))

            with Timer(timer_name) as _:
                with open(compressed_path, "rb") as infile:
                    decompressed_bytes = brotli.decompress(infile.read())

        assert input_bytes == decompressed_bytes
        print_size(compressed_path, len(input_bytes), timer_name)


def main():
    args = parse_args()
    benchmark_gzip(args.target, 6, "gzip_6 (baseline)")
    benchmark_gzip(args.target, 1, "gzip_1")
    benchmark_gzip(args.target, 9, "gzip_9")

    benchmark_lz4(args.target, 1, "lz4_1")
    benchmark_lz4(args.target, 9, "lz4_9")

    benchmark_zstd(args.target, -10, "zstd_minus10")
    benchmark_zstd(args.target, 1, "zstd_1")
    benchmark_zstd(args.target, 9, "zstd_9")
    benchmark_zstd(args.target, 20, "zstd_20")

    benchmark_snappy(args.target, "snappy")

    benchmark_brotli(args.target, 1, "brotli_1")
    benchmark_brotli(args.target, 9, "brotli_9")

    print("")
    print("TIMINGS [s]")
    print_timers()


if __name__ == "__main__":
    main()

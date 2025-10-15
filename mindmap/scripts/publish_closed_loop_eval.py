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
import glob
import json
import os
import pathlib
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Publish eval result as html")
    parser.add_argument(
        "--eval_file_path", type=pathlib.Path, required=True, help="Path to eval results"
    )
    parser.add_argument(
        "--videos_path",
        type=pathlib.Path,
        required=True,
        help="Path to dir containing generated mp4 files",
    )
    parser.add_argument("--output_path", type=pathlib.Path, required=True, help="Output dir")

    return parser.parse_args()


def main():
    args = parse_args()

    videos_output_path = os.path.join(args.output_path, "videos")
    os.makedirs(videos_output_path, exist_ok=True)

    create_html(videos_output_path, args.eval_file_path, args.output_path)

    convert_videos_to_browser_friendy_format(args.videos_path, videos_output_path)

    return 0


def create_html(videos_output_path, eval_file_path, output_path):
    """Write a html file with results and video"""
    with open(eval_file_path, "r") as file:
        data = json.load(file)

    TABLE_HEADER = "<table><tr><th>Name</th><th>Video</th><th>Eval</th></tr>"

    HTML_HEAD = """<html>
    <head>
    <style>
    table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        font-size: 16px;
        background-color: white;
    }

    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }

    th {
        background-color: #007BFF;
        color: white;
    }

    tr:nth-child(even) {
        background-color: #f2f2f2;
    }

    tr:hover {
        background-color: #cce5ff;
        transition: 0.3s;
    }
    </style>
    </head>
    <body>
    """

    # Build html strings for failed and successful demos separately
    html_success = ""
    html_failed = ""
    for key, value in sorted(data.items()):
        if "demo" in key:
            html_demo = html_table_entry(value["demo"], videos_output_path, value)
            if value["success"] == True:
                html_success += html_demo
            else:
                html_failed += html_demo

    # Create the html document
    html = HTML_HEAD

    html += html_summary(data)

    html += TABLE_HEADER
    html += "<h2>Failed demos</h2>"
    html += html_failed
    html += "</table>"

    html += TABLE_HEADER
    html += "<h2>Successful demos</h2>"
    html += html_success
    html += "</table>"

    html += "</body></html>"

    # Save to an HTML file
    output_html = os.path.join(output_path, "index.html")
    with open(output_html, "w") as f:
        f.write(html)

    print(f"Wrote {output_html}")


def html_summary(data):
    summary = data["summary"]
    metadata = data["metadata"]

    num_success = summary["success"]
    success_rate = summary["success_rate"]

    html = "<h1>Closed loop evaluation</h1>"
    for key, value in metadata.items():
        html += f"<p><b>{key}:</b> {value}</p>"
    html += f"<p><b>success rate:</b> {success_rate}"

    return html


def html_table_entry(demo_name, videos_output_path, stats):
    """Return html for one entry in the result table"""
    video_relative_url = f"videos/{demo_name}.mp4"

    html = f"<tr>"
    html += f"<td>{demo_name}</td>"
    html += f"<td>" + html_video(video_relative_url) + "</td>"
    html += "<td>" + " ".join(f"{k}: {v}<br>" for k, v in stats.items()) + "</td>"
    html += f"</tr>"
    return html


def html_video(filename):
    """Return html that embeds a video"""
    html = f"""
<video width="240" height="240" controls>
    <source src="{filename}" type="video/mp4">
    Your browser does not support the video tag.
</video>
"""
    return html


def convert_videos_to_browser_friendy_format(path, output_path):
    video_files = glob.glob(os.path.join(path, "*.mp4"))
    assert len(video_files) > 0, f"No mp4 files found in {path}"

    for video_file in video_files:
        output_file = os.path.join(output_path, os.path.basename(video_file))
        print(f"Creating {output_path}")
        subprocess.run(
            f"ffmpeg -i {video_file} -c:v libx264 -crf 23 -preset fast -c:a aac -b:a 128k {output_file}".split(),
            check=True,
        )
    subprocess.run


if __name__ == "__main__":
    sys.exit(main())

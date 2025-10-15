# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import abc
import argparse
import enum
import json
import os
import pathlib
import subprocess
import time
from urllib.parse import quote

from bs4 import BeautifulSoup
import requests

# Directory on the remote where our checkpoints are stored.
OSMO_CHECKPOINTS_REMOTE_DIR = "osmo/data/output/train_logs/checkpoints"
FILES_TO_DOWNLOAD = ["best.pth", "last.pth", "training_args.json"]


class DownloadMethod(enum.Enum):
    # Download checkpoints from PDX storage. This only works if training task is completed
    PDX = "pdx"
    # Download checkpoints from FileBrowser. This can be used to download from an ongoing training task
    FILEBROWSER = "filebrowser"


class FileApi(abc.ABC):
    @abc.abstractmethod
    def list_directory(self, remote_path: pathlib.Path) -> list[pathlib.Path]:
        pass

    @abc.abstractmethod
    def download_file(self, remote_path: pathlib.Path, local_path: pathlib.Path) -> None:
        pass


class FileBrowserFileApi(FileApi):
    def __init__(self, server_url: str, username: str, password: str, workflow: str):
        self.server_url = server_url.rstrip("/")
        self.username = username
        self.password = password
        self._port_process = start_port_forwarding(workflow)
        for _ in range(10):
            try:
                time.sleep(1)
                self._token = self._connect()
                break
            except requests.exceptions.ConnectionError:
                pass
        if not hasattr(self, "_token"):
            raise RuntimeError("Failed to connect to running OSMO workflow.")

    def __del__(self):
        self.end_port_forwarding()

    def _connect(self) -> str:
        print(f"Trying to connect to {self.server_url} as {self.username}")
        login_url = f"{self.server_url}/api/login"
        response = requests.post(
            login_url, json={"username": self.username, "password": self.password}
        )
        response.raise_for_status()
        print("Connected.")
        return response.text

    def list_directory(self, remote_path: pathlib.Path) -> list[pathlib.Path]:
        encoded_path = quote(str(remote_path))
        # Need to go over the API endpoint for FileBrowser
        url = f"{self.server_url}/api/resources/{encoded_path}"
        response = requests.get(url, headers={"X-Auth": self._token})
        response.raise_for_status()
        return [pathlib.Path(item["path"]) for item in response.json().get("items", [])]

    def download_file(self, remote_path: pathlib.Path, local_path: pathlib.Path) -> None:
        print(f"Downloading {remote_path} to {local_path}.")
        encoded_path = quote(str(remote_path))
        url = f"{self.server_url}/api/raw/{encoded_path}"
        response = requests.get(url, headers={"X-Auth": self._token})
        response.raise_for_status()

        filename = str(remote_path).split("/")[-1]
        local_path = pathlib.Path(local_path) / filename
        local_path.write_bytes(response.content)

    def end_port_forwarding(self):
        print("Terminating port forwarding process.")
        self._port_process.terminate()
        self._port_process.wait()


def start_port_forwarding(workflow: str) -> subprocess.Popen:
    process = subprocess.Popen(
        ["osmo", "workflow", "port-forward", workflow, "training", "--port", "8080:8080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process


def get_workflow_status(workflow: str) -> str:
    result = subprocess.run(
        ["osmo", "workflow", "query", workflow, "--format-type", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to query workflow status: {result.stderr}")
    return json.loads(result.stdout).get("status", "UNKNOWN")


def get_pdx_checkpoint_url(workflow_name: str) -> str:
    """Get the URL for downloading checkpoints from PDX storage.

    This function constructs the URL for a workflow's checkpoint directory in PDX storage,
    makes a request to list the contents, and parses the HTML response to find the specific
    checkpoint subdirectory.

    Args:
        workflow_name: Name of the OSMO workflow to get checkpoints for

    Returns:
        str: Full URL to the checkpoint directory containing model files
    """
    # Get the checkpoint directory
    base_url = f"PLACEHOLDER_URL/{workflow_name}"

    response = requests.get(base_url, allow_redirects=True)
    if not response.ok:
        raise RuntimeError(
            f"Failed to get checkpoint directory: {response.status_code} - {response.text}"
        )

    # Parse HTML to find the directory name
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all directory links (they end with /)
    dir_links = [a for a in soup.find_all("a") if a["href"].endswith("/") and a["href"] != "../"]

    if not dir_links:
        raise RuntimeError("No checkpoint directories found in the response")

    if len(dir_links) > 1:
        raise RuntimeError(
            f"Found multiple checkpoint directories: {[link['href'] for link in dir_links]}"
        )

    subdir_name = dir_links[0]["href"].rstrip("/")
    remote_output_dir = f"{base_url}/{subdir_name}"
    return remote_output_dir


def download_from_pdx(workflow_name: str, local_output_dir: str):
    remote_output_dir = get_pdx_checkpoint_url(workflow_name)

    for filename in FILES_TO_DOWNLOAD:
        local_file_path = os.path.join(local_output_dir, filename)
        print(f"Downloading {remote_output_dir}/{filename} to {local_file_path}.")
        try:
            subprocess.run(
                ["wget", "-O", str(local_file_path), os.path.join(remote_output_dir, filename)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download {filename}: {e}")


def download_from_filebrowser(workflow_name: str, local_output_dir: str):
    file_api = FileBrowserFileApi(
        server_url="http://localhost:8080",
        username="PLACEHOLDER_USERNAME",
        password="PLACEHOLDER_PASSWORD",
        workflow=workflow_name,
    )
    remote_output_dir = file_api.list_directory(pathlib.Path(OSMO_CHECKPOINTS_REMOTE_DIR))[0]

    for filename in FILES_TO_DOWNLOAD:
        file_api.download_file(
            remote_output_dir / filename, pathlib.Path(local_output_dir) / filename
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI tool to download checkpoints from multiple running OSMO jobs"
    )
    parser.add_argument(
        "-w",
        "--workflow_names",
        type=str,
        nargs="+",
        required=True,
        help="List of Osmo workflow names for which to download the checkpoints.",
    )
    parser.add_argument(
        "-d",
        "--download_base_dir",
        type=pathlib.Path,
        required=True,
        help="Local directory to download the checkpoints to. A directory for each"
        "workflow name will be generated to store the corresponding checkpoints",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=DownloadMethod,
        default=DownloadMethod.FILEBROWSER,
        help="Method to download checkpoints.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    for workflow_name in args.workflow_names:
        print(f"Downloading checkpoints for workflow: {workflow_name}")
        try:
            local_output_dir = os.path.join(args.download_base_dir, workflow_name)
            os.makedirs(local_output_dir, exist_ok=True)
            if args.method == DownloadMethod.PDX:
                download_from_pdx(workflow_name, local_output_dir)
            elif args.method == DownloadMethod.FILEBROWSER:
                download_from_filebrowser(workflow_name, local_output_dir)
            else:
                raise ValueError(f"Invalid download method: {args.method}")
            print(f"Successfully downloaded checkpoints for {workflow_name}")
        except Exception as e:
            print(f"Failed to download checkpoints for {workflow_name}: {e}")
            raise e


if __name__ == "__main__":
    main()

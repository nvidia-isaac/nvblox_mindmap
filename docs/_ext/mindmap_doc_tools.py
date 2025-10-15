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
from typing import Any, List

from sphinx.application import Sphinx


def mindmap_git_url(app: Sphinx, _: Any, source: List[str]) -> None:
    """Replaces the :mindmap_git_url: directive with a URL string.

    The output git clone command depends on whether we're in release or internal mode.

    """

    def replacer(_: Any) -> str:
        release_state = app.config.mindmap_docs_config["released"]
        internal_git_url = app.config.mindmap_docs_config["internal_git_url"]
        external_git_url = app.config.mindmap_docs_config["external_git_url"]
        if release_state:
            git_clone_target = external_git_url
        else:
            git_clone_target = internal_git_url
        return str(git_clone_target)

    source[0] = re.sub(r":mindmap_git_url:", replacer, source[0])


def mindmap_repo_link(app: Sphinx, _: Any, source: List[str]) -> None:
    """Replaces the :mindmap_repo_link: directive with a repo link.

    The output link is either gitlab (internal) or github (external) depending on the release state.

    """

    def replacer(match: re.Match) -> str:
        link_name = match.group("link_name")
        release_state = app.config.mindmap_docs_config["released"]
        internal_code_link_base_url = app.config.mindmap_docs_config["internal_code_link_base_url"]
        external_code_link_base_url = app.config.mindmap_docs_config["external_code_link_base_url"]
        if release_state:
            code_link_base_url = external_code_link_base_url
        else:
            code_link_base_url = internal_code_link_base_url
        return f"`{link_name} <{code_link_base_url}>`_"

    source[0] = re.sub(r":mindmap_repo_link:`<(?P<link_name>.*)>`", replacer, source[0])


def mindmap_code_link(app: Sphinx, _: Any, source: List[str]) -> None:
    """Replaces the :mindmap_code_link: directive with a code block.

    The output link is either gitlab (internal) or github (external) depending on the release state.
    Supports optional line number anchors, e.g. :mindmap_code_link:`<mindmap/cli/args.py#L212>`.

    """

    def replacer(match: re.Match) -> str:
        relative_path = match.group("relative_path")
        release_state = app.config.mindmap_docs_config["released"]
        internal_code_link_base_url = app.config.mindmap_docs_config["internal_code_link_base_url"]
        external_code_link_base_url = app.config.mindmap_docs_config["external_code_link_base_url"]
        # Extract the file name
        file_name = relative_path.split("/")[-1]
        if release_state:
            code_link_base_url = external_code_link_base_url
        else:
            code_link_base_url = internal_code_link_base_url
        return f"`{file_name} <{code_link_base_url}/{relative_path}>`_"

    source[0] = re.sub(r":mindmap_code_link:`<(?P<relative_path>.*)>`", replacer, source[0])


def setup(app: Sphinx) -> None:
    app.connect("source-read", mindmap_git_url)
    app.connect("source-read", mindmap_code_link)
    app.connect("source-read", mindmap_repo_link)
    app.add_config_value("mindmap_docs_config", {}, "env")

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# pylint: disable=redefined-builtin

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
from typing import List

# Modify PYTHONPATH so we can obtain the version data from setup module.
# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from setup import MINDMAP_VERSION_NUMBER

# Modify PYTHONPATH so we can import the helpers module.
sys.path.insert(0, os.path.abspath("."))
from helpers import TemporaryLinkcheckIgnore, is_expired, to_datetime

# -- Project information -----------------------------------------------------

project = "mindmap"
copyright = "2025, NVIDIA"
author = "NVIDIA"
released = True  # Indicates if this is a public or internal version of the repo.

# -- General configuration ---------------------------------------------------

sys.path.append(os.path.abspath("_ext"))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx_tabs.tabs",
    "sphinx_copybutton",
    "sphinx_substitution_extensions",
    "mindmap_doc_tools",
]

# put type hints inside the description instead of the signature (easier to read)
autodoc_typehints = "description"
# document class *and* __init__ methods
autoclass_content = "both"  #

todo_include_todos = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*/site-packages/*", "core_docs_venv/*"]

# Be picky about missing references
nitpicky = True  # warns on broken references
nitpick_ignore: List[str] = []  # can exclude known bad refs

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "nvidia_sphinx_theme"
html_title = f"mindmap {MINDMAP_VERSION_NUMBER}"
html_show_sphinx = False
html_theme_options = {
    "copyright_override": {"start": 2023},
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "footer_links": {},
    "github_url": "https://github.com/NVlabs/nvblox_mindmap",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = []
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Todos
todo_include_todos = True

# Linkcheck
# NOTE(remos): We are currently ignoring private huggingface/github links during linkcheck.
# Otherwise linkcheck will fail with unauthorized error.
# TODO(remos): Remove this once we pushed the huggingface datasets and github repo to public.
linkcheck_ignore = [r"https://huggingface.co*", r"https://github.com/NVlabs/nvblox_mindmap*"]

#####################################
#  Macros dependent on release state
#####################################

mindmap_docs_config = {
    "released": released,
    "internal_git_url": "ssh://git@gitlab-master.nvidia.com:12051/nvblox/mindmap.git",
    "external_git_url": "git@github.com:NVlabs/nvblox_mindmap.git",
    "internal_code_link_base_url": "https://gitlab-master.nvidia.com/nvblox/mindmap/-/tree/main",
    "external_code_link_base_url": "https://github.com/NVlabs/nvblox_mindmap/tree/public",
}

rst_prolog = """
.. |cube_stacking_hdf5| replace:: mindmap_franka_cube_stacking_1000_demos.hdf5
.. |mug_in_drawer_hdf5| replace:: mindmap_franka_mug_in_drawer_250_demos.hdf5
.. |drill_in_box_hdf5| replace:: mindmap_gr1_drill_in_box_200_demos.hdf5
.. |stick_in_bin_hdf5| replace:: mindmap_gr1_stick_in_bin_200_demos.hdf5
.. |cube_stacking_HF_dataset| replace:: nvidia/PhysicalAI-Robotics-mindmap-Franka-Cube-Stacking
.. |mug_in_drawer_HF_dataset| replace:: nvidia/PhysicalAI-Robotics-mindmap-Franka-Mug-in-Drawer
.. |drill_in_box_HF_dataset| replace:: nvidia/PhysicalAI-Robotics-mindmap-GR1-Drill-in-Box
.. |stick_in_bin_HF_dataset| replace:: nvidia/PhysicalAI-Robotics-mindmap-GR1-Stick-in-Bin
.. |mindmap_checkpoints_HF| replace:: nvidia/PhysicalAI-Robotics-mindmap-Checkpoints
"""

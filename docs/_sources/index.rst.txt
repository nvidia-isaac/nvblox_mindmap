``mindmap`` Overview
====================

.. list-table::
    :class: gallery

    * - .. figure:: images/teaser_isaaclab.jpg
         :height: 350px
         :width: 350px

         Humanoid robot performing the ``Drill in Box`` spatial memory task in Isaac Lab simulation.
      - .. figure:: images/teaser_reconstruction.png
         :height: 350px
         :width: 350px

         Deep feature map (colored by PCA) that enables spatial memory for the ``mindmap`` 3D action policy.

What is mindmap?
----------------

mindmap is a 3D diffusion policy that generates robot trajectories based on semantic 3D reconstruction of the environment.
Unlike traditional approaches that rely solely on current visual input,
mindmap maintains spatial memory through deep feature maps,
enabling robots to perform complex manipulation tasks that require remembering the layout of the environment and the locations of objects.

Our :ref:`mindmap paper <Papers>` demonstrates that mindmap effectively solves manipulation
tasks where state-of-the-art approaches without memory mechanisms struggle. Through extensive simulation experiments, we show significant improvements in task success rates for scenarios requiring spatial memory.

mindmap builds upon the `nvblox codebase <https://github.com/nvidia-isaac/nvblox/tree/public>`_,
which provides a GPU-accelerated reconstruction system wrapped in PyTorch. This system is essential for building the deep feature maps that enable spatial memory in mindmap.


Open Source Release
-------------------

We release our complete :mindmap_repo_link:`<codebase>` including:

- **Reconstruction System**: Real-time 3D mapping and feature extraction
- **Training Code**: End-to-end model training pipeline
- **Evaluation Tasks**: Four benchmark tasks for spatial memory testing
- **Pre-trained Models**: Ready-to-use checkpoints for immediate evaluation

.. note::

   This project will download and install additional third-party open source software projects.
   Review the license terms of these open source projects before use.

System Requirements
-------------------

- **Architecture**: ``mindmap`` is only supported on x86 architecture.
- **OS**: The ``mindmap`` container is only supported on Linux.
- **GPU**: We recommend using a discrete NVIDIA GPU with at least 16GB of VRAM.
- **Simulation**: ``mindmap`` is using `Isaac Lab 2.1.0` / `Isaac Sim 4.5.0` for simulation. Make sure your system meets the `simulation requirements <https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html#system-requirements>`_.

Getting Started
---------------

To begin using mindmap:

#. Follow the setup instructions in :doc:`pages/setup`
#. Choose a workflow from :doc:`pages/example_workflows`
#. Explore the available tasks in :doc:`pages/mindmap_tasks`

License
-------

This project is licensed under the ``NVIDIA License (NSCLv1)``. See :mindmap_code_link:`<LICENSE.md>` for details.

.. _Papers:

Papers
------

If you find mindmap useful for your research, please consider citing our work:

* Remo Steiner, Alexander Millane, David Tingdahl, Clemens Volk, Vikram Ramasamy, Xinjie Yao, Peter Du, Soha Pouya and Shiwei Sheng.
  "**mindmap: Spatial Memory in Deep Feature Maps for 3D Action Policies**". CoRL 2025 Workshop RemembeRL.
  `arXiv preprint arXiv:2509.20297 (2025). <https://arxiv.org/abs/2509.20297>`_
* Alexander Millane, Helen Oleynikova, Emilie Wirbel, Remo Steiner, Vikram Ramasamy, David Tingdahl, and Roland Siegwart.
  "**nvblox: GPU-Accelerated Incremental Signed Distance Field Mapping**". ICRA 2024.
  `arXiv preprint arXiv:2311.00626 (2024). <https://arxiv.org/abs/2311.00626>`_

Documentation Structure
-----------------------

.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   pages/setup
   pages/example_workflows
   mindmap Tasks <pages/mindmap_tasks>

.. toctree::
   :maxdepth: 1
   :caption: Data Generation

   pages/create_task
   pages/demo_recording
   pages/data_generation

.. toctree::
   :maxdepth: 1
   :caption: Training

   pages/training

.. toctree::
   :maxdepth: 1
   :caption: Evaluation

   pages/open_loop_evaluation
   pages/closed_loop_evaluation

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   pages/parameters

.. toctree::
   :maxdepth: 1
   :caption: Downloads

   pages/download_datasets
   pages/download_models

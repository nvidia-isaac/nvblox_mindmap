Setting up mindmap
==================

Follow these steps to set up mindmap on your system:

#. **Prerequisites**

   * Install NVIDIA Container Toolkit by following the `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_.

   * Depending on your docker version, you might need to install the `docker buildkit extensions <https://docs.docker.com/build/buildkit/>`_:

   .. code-block:: bash

       sudo apt install docker-buildx


#. **Clone the mindmap repository**

   .. code-block:: bash

      git clone :mindmap_git_url:

#. **Initialize submodules**

   Initialize mindmap's required submodules (nvblox and Isaac Lab):

   .. code-block:: bash

      git submodule update --init --recursive

#. **Launch the Docker container**

   Build and launch an interactive Docker container with all dependencies pre-installed:

   .. code-block:: bash

      ./docker/run_docker.sh

.. note::

   The Docker run script will download and install additional third-party open source software projects.
   Review the license terms of these open source projects before use.

#. **Get started**

   Choose a workflow from :doc:`Example Workflows <example_workflows>` to begin using mindmap.

.. note::

   All dependencies (including IsaacSim and Isaac Lab) required for running mindmap
   are pre-installed inside the Docker container. No additional installations are needed.

.. note::

   The Docker container automatically mounts three directories from your host machine's home folder
   (``~/datasets``, ``~/models``, and ``~/eval``) to the corresponding directories in the container
   (``/datasets``, ``/models``, and ``/eval``). This allows you to access your datasets, models,
   and evaluation results from both the host machine and the Docker container seamlessly.

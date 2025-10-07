Generating a Dataset
====================

This page explains how to generate a mindmap dataset from HDF5 demonstration files.
The data generation process converts raw demonstrations into the format required for training mindmap models.

Prerequisites
-------------

#. .. include:: ../snippets/installation_required_snippet.rst

#. Obtain HDF5 demonstration files by either:

   - :doc:`Downloading pre-generated datasets <download_datasets>`, or
   - :doc:`Recording your own demonstrations <demo_recording>`


Dataset Generation
------------------

Generate a mindmap dataset for your chosen task:

.. tabs::
    .. tab:: Cube Stacking

        .. code-block:: bash
            :substitutions:

            python run_isaaclab_datagen.py \
                --task cube_stacking \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --output_dir <OUTPUT_DATASET_PATH> \
                --demos_datagen 0-9 \
                --hdf5_file <HDF5_DATASET_PATH>/|cube_stacking_hdf5|

    .. tab:: Mug in Drawer

        .. code-block:: bash
            :substitutions:

            python run_isaaclab_datagen.py \
                --task mug_in_drawer \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --output_dir <OUTPUT_DATASET_PATH> \
                --demos_datagen 0-9 \
                --hdf5_file <HDF5_DATASET_PATH>/|mug_in_drawer_hdf5|

    .. tab:: Drill in Box

        .. code-block:: bash
            :substitutions:

            python run_isaaclab_datagen.py \
                --task drill_in_box \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --output_dir <OUTPUT_DATASET_PATH> \
                --demos_datagen 0-9 \
                --hdf5_file <HDF5_DATASET_PATH>/|drill_in_box_hdf5|

    .. tab:: Stick in Bin

        .. code-block:: bash
            :substitutions:

            python run_isaaclab_datagen.py \
                --task stick_in_bin \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --output_dir <OUTPUT_DATASET_PATH> \
                --demos_datagen 0-9 \
                --hdf5_file <HDF5_DATASET_PATH>/|stick_in_bin_hdf5|

.. note::

   Replace ``<HDF5_DATASET_PATH>`` with the path containing your input HDF5 file
   and ``<OUTPUT_DATASET_PATH>`` with your desired output directory for the generated dataset.

.. include:: ../snippets/parameter_note_snippet.rst

.. _mindmap_dataset_structure:

Dataset Structure
-----------------

The generated dataset follows this structure:

.. code-block:: text

   📂 <OUTPUT_DATASET_PATH>/
   ├── 📂 demo_00000/
   │   ├── 00000.<CAMERA_NAME>_depth.png
   │   ├── 00000.<CAMERA_NAME>_intrinsics.npy
   │   ├── 00000.<CAMERA_NAME>_pose.npy
   │   ├── 00000.<CAMERA_NAME>_rgb.png
   │   ├── 00000.nvblox_vertex_features.zst
   │   ├── 00000.robot_state.npy
   │   ├── ...
   │   ├── <NUM_STEPS_IN_DEMO>.<CAMERA_NAME>_depth.png
   │   ├── ...
   │   ├── <NUM_STEPS_IN_DEMO>.robot_state.npy
   │   └── demo_successful.npy
   ├── 📂 demo_00001/
   │   └── ...
   └── 📂 demo_<NUMBER_OF_DEMOS>/

The dataset contents depend on the ``--data_type`` parameter:

- ``RGBD``: Contains only camera files
- ``MESH``: Contains only vertex features
- ``RGBD_AND_MESH``: Contains both camera files and vertex features

Robot state data (end-effector pose, gripper state, and optionally head yaw) is available for all data types.

The camera naming convention varies by task type:

- ``<CAMERA_NAME>`` is `wrist` for robot arm tasks (``Cube Stacking`` and ``Mug in Drawer``)
- ``<CAMERA_NAME>`` is `pov` for humanoid robot tasks (``Drill in Box`` and ``Stick in Bin``)

When using the ``--add_external_cam`` flag, an additional camera is included:

- `external` camera for humanoid tasks
- `table` camera for robot arm tasks

With the external camera enabled, the reconstruction in ``nvblox_vertex_features.zst`` uses RGBD data from both cameras.
By default, only the RGBD data from the primary camera (`pov` or `wrist`) is used.

For more details on configuration options, see :doc:`parameters` and refer to :doc:`open_loop_evaluation` for dataset visualization instructions.

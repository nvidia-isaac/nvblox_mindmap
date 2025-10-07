Open Loop Evaluation
====================

Open loop evaluation allows you to visualize datasets and test model predictions on recorded demonstration data.
This is useful for debugging, understanding model behavior, and validating performance before closed loop testing.

Prerequisites
-------------

#. .. include:: ../snippets/installation_required_snippet.rst

#. Obtain a dataset by either:

   - :doc:`Downloading a pre-generated dataset <download_datasets>`, or
   - :doc:`Generating your own dataset <data_generation>`

#. Obtain a trained model by either:

   - :doc:`Downloading a pre-trained checkpoint <download_models>`, or
   - :doc:`Training your own model <training>`

Running Open Loop Evaluation
----------------------------

Run open loop evaluation for your chosen task:

.. tabs::
    .. tab:: Cube Stacking

        .. code-block:: bash
            :substitutions:

            torchrun_local run_open_loop_policy.py \
                --task cube_stacking \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --checkpoint <LOCAL_CHECKPOINT_PATH>/best.pth \
                --demos_open_loop 0 \
                --dataset <LOCAL_DATASET_PATH>

    .. tab:: Mug in Drawer

        .. code-block:: bash
            :substitutions:

            torchrun_local run_open_loop_policy.py \
                --task mug_in_drawer \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --checkpoint <LOCAL_CHECKPOINT_PATH>/best.pth \
                --demos_open_loop 0 \
                --dataset <LOCAL_DATASET_PATH>

    .. tab:: Drill in Box

        .. code-block:: bash
            :substitutions:

            torchrun_local run_open_loop_policy.py \
                --task drill_in_box \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --checkpoint <LOCAL_CHECKPOINT_PATH>/best.pth \
                --demos_open_loop 0 \
                --dataset <LOCAL_DATASET_PATH>

    .. tab:: Stick in Bin

        .. code-block:: bash
            :substitutions:

            torchrun_local run_open_loop_policy.py \
                --task stick_in_bin \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --checkpoint <LOCAL_CHECKPOINT_PATH>/best.pth \
                --demos_open_loop 0 \
                --dataset <LOCAL_DATASET_PATH>

.. note::

   Update the ``--demos_open_loop`` argument to visualize different demonstrations.
   Make sure you have the corresponding demo in the dataset.

.. note::

   Replace ``<LOCAL_DATASET_PATH>`` with your dataset directory path
   and ``<LOCAL_CHECKPOINT_PATH>`` with your checkpoint directory path.

.. note::

   To visualize the dataset without running model inference, simply omit the ``--checkpoint`` argument.

.. note::

   The ``--only_sample_keyposes`` argument can be used to only run predictions and visualizations for the keyposes of the dataset.
   This is useful to avoid having to step through every sample of the dataset.

Interactive Visualization
-------------------------

After running the command, multiple visualization windows will open (see :ref:`open_loop_visualizations` below). To navigate through the data:

1. **Select a 3D window** by clicking on it
2. **Press space** to iterate through the dataset
3. **Observe** the model's predictions and spatial memory reconstruction

.. _open_loop_visualizations:

Visualization Examples
----------------------

The following examples show the expected visualizations for each task:

.. list-table::
    :class: gallery

    * - .. figure:: ../images/open_loop_cube_stacking_pov.png
         :height: 250px

         Cube Stacking
      - .. figure:: ../images/open_loop_cube_stacking.png
         :height: 300px

    * - .. figure:: ../images/open_loop_mug_in_drawer_pov.png
         :height: 250px

         Mug in Drawer
      - .. figure:: ../images/open_loop_mug_in_drawer.png
         :height: 300px

    * - .. figure:: ../images/open_loop_drill_in_box_pov.png
         :height: 250px

         Drill in Box
      - .. figure:: ../images/open_loop_drill_in_box.png
         :height: 300px

    * - .. figure:: ../images/open_loop_stick_in_bin_pov.png
         :height: 250px

         Stick in Bin
      - .. figure:: ../images/open_loop_stick_in_bin.png
         :height: 300px

On the left column, the RGB ego-camera view is shown.

On the right column, the ``Model Inputs and Prediction`` window shows the 3D visualization of the model's inputs and prediction:

- **World Axis**: The global coordinate frame
- **Keypose History**: Previous ``num_history`` keyposes (model input) labeled as ``keypose i-<history_idx> gripper<gripper_idx>``
- **Current RGBD View**: Current camera view as a colored point cloud
- **Feature Map**: 3D reconstruction as a vertex feature point cloud (colored by PCA)
- **Keypose Prediction**: Next ``prediction_horizon`` keyposes predicted by the model (labeled as ``predicted i+<prediction_idx> gripper<gripper_idx>``) - only shown when using a checkpoint

This visualization is useful for debugging model performance by showing how spatial information is reconstructed
and how action predictions are generated from the model inputs.

.. include:: ../snippets/parameter_note_snippet.rst

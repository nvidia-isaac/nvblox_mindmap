Closed Loop Evaluation
======================

Closed loop evaluation tests a trained mindmap model in Isaac Lab simulation.
Observational data is fed to the model in real-time, and the model's actions are executed in the simulation.

Task Demonstrations
-------------------

The following GIFs show mindmap models successfully completing each benchmark task:

.. list-table::
    :class: gallery

    * - .. figure:: ../images/closed_loop_cube_stacking.gif
         :width: 250px

         Cube Stacking
      - .. figure:: ../images/closed_loop_mug_in_drawer.gif
         :width: 250px

         Mug in Drawer
    * - .. figure:: ../images/closed_loop_drill_in_box.gif
         :width: 250px

         Drill in Box
      - .. figure:: ../images/closed_loop_stick_in_bin.gif
         :width: 250px

         Stick in Bin

Prerequisites
-------------

#. .. include:: ../snippets/installation_required_snippet.rst

#. Obtain HDF5 demonstration files by either:

   - :doc:`Downloading a pre-generated dataset <download_datasets>`, or
   - :doc:`Recording your own demonstrations <demo_recording>`

#. Obtain a trained model by either:

   - :doc:`Downloading a pre-trained checkpoint <download_models>`, or
   - :doc:`Training your own model <training>`


Running Closed Loop Evaluation
------------------------------

Evaluate your model on the chosen task:

.. tabs::
    .. tab:: Cube Stacking

        .. code-block:: bash
            :substitutions:

            torchrun_local run_closed_loop_policy.py \
                --task cube_stacking \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --checkpoint <LOCAL_CHECKPOINT_PATH>/best.pth \
                --demos_closed_loop 150-249 \
                --hdf5_file <LOCAL_DATASET_PATH>/|cube_stacking_hdf5|

    .. tab:: Mug in Drawer

        .. code-block:: bash
            :substitutions:

            torchrun_local run_closed_loop_policy.py \
                --task mug_in_drawer \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --checkpoint <LOCAL_CHECKPOINT_PATH>/best.pth \
                --demos_closed_loop 150-249 \
                --hdf5_file <LOCAL_DATASET_PATH>/|mug_in_drawer_hdf5|

    .. tab:: Drill in Box

        .. code-block:: bash
            :substitutions:

            torchrun_local run_closed_loop_policy.py \
                --task drill_in_box \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --checkpoint <LOCAL_CHECKPOINT_PATH>/best.pth \
                --demos_closed_loop 100-199 \
                --hdf5_file <LOCAL_DATASET_PATH>/|drill_in_box_hdf5|

    .. tab:: Stick in Bin

        .. code-block:: bash
            :substitutions:

            torchrun_local run_closed_loop_policy.py \
                --task stick_in_bin \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --checkpoint <LOCAL_CHECKPOINT_PATH>/best.pth \
                --demos_closed_loop 100-199 \
                --hdf5_file <LOCAL_DATASET_PATH>/|stick_in_bin_hdf5|

.. note::

    The selected demos in the commands above correspond to 100 demonstrations each from the evaluation set of the :doc:`pre-trained models <download_models>`.

.. note::

    Using the ``--record_videos`` flag, closed loop evaluation runs can be recorded and stored at the path specified with ``--record_camera_output_path <VIDEO_OUTPUT_DIR>``.

Alternatively, you can evaluate a task in ground truth mode, which replays the ground truth keyposes from a dataset:

.. tabs::
    .. tab:: Cube Stacking

        .. code-block:: bash
            :substitutions:

            torchrun_local run_closed_loop_policy.py \
                --task cube_stacking \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --dataset <LOCAL_DATASET_PATH> \
                --demos_closed_loop 0-9 \
                --hdf5_file <LOCAL_DATASET_PATH>/|cube_stacking_hdf5| \
                --demo_mode execute_gt_goals

    .. tab:: Mug in Drawer

        .. code-block:: bash
            :substitutions:

            torchrun_local run_closed_loop_policy.py \
                --task mug_in_drawer \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --dataset <LOCAL_DATASET_PATH> \
                --demos_closed_loop 0-9 \
                --hdf5_file <LOCAL_DATASET_PATH>/|mug_in_drawer_hdf5| \
                --demo_mode execute_gt_goals

    .. tab:: Drill in Box

        .. code-block:: bash
            :substitutions:

            torchrun_local run_closed_loop_policy.py \
                --task drill_in_box \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --dataset <LOCAL_DATASET_PATH> \
                --demos_closed_loop 0-9 \
                --hdf5_file <LOCAL_DATASET_PATH>/|drill_in_box_hdf5| \
                --demo_mode execute_gt_goals

    .. tab:: Stick in Bin

        .. code-block:: bash
            :substitutions:

            torchrun_local run_closed_loop_policy.py \
                --task stick_in_bin \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --dataset <LOCAL_DATASET_PATH> \
                --demos_closed_loop 0-9 \
                --hdf5_file <LOCAL_DATASET_PATH>/|stick_in_bin_hdf5| \
                --demo_mode execute_gt_goals

Running in ground truth mode is useful for validating the keypose extraction pipeline and estimating the model's maximum achievable performance before training.

Evaluation Results
------------------

After completing all selected demonstrations, check the model's success rate:

- **Console output**: Look for the lines after ``Summary of closed loop evaluation``
- **Evaluation file**: Check the file specified by ``--eval_file_path`` (if provided)

The success rate indicates how many demonstrations the model completed successfully.

.. note::

    Closed loop evaluation is not deterministic,
    i.e. the same demonstration can succeed or fail on different runs even if the same model or ground truth goals are used.
    Therefore, it is important to run enough demonstrations to get a statistically significant result.

Visualization Options
---------------------

To visualize model inputs and outputs during evaluation:

1. **Add the** ``--visualize`` **flag** to your command
2. **Select a visualization window** by clicking on it
3. **Press space** to trigger the next inference step

This allows you to observe how the model processes spatial information and makes real-time predictions.

.. note::

   Replace ``<LOCAL_DATASET_PATH>`` with your dataset directory path
   and ``<LOCAL_CHECKPOINT_PATH>`` with your checkpoint directory path.

.. include:: ../snippets/parameter_note_snippet.rst

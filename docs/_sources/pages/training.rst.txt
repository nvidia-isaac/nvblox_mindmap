Training a Model
================

This page explains how to train a mindmap model using your generated or downloaded dataset. The training process learns spatial memory representations from demonstration data.

Prerequisites
-------------

#. .. include:: ../snippets/installation_required_snippet.rst

#. Obtain a dataset by either:

   - :doc:`Downloading a pre-generated dataset <download_datasets>`, or
   - :doc:`Generating your own dataset <data_generation>`

Training Process
----------------

Train a mindmap model for your chosen task:

.. tabs::
    .. tab:: Cube Stacking

        .. code-block:: bash
            :substitutions:

            torchrun_local run_training.py \
                --task cube_stacking \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --demos_train 0-6 \
                --demos_valset 7-9 \
                --dataset <LOCAL_DATASET_PATH> \
                --base_log_dir <OUTPUT_DIR>

    .. tab:: Mug in Drawer

        .. code-block:: bash
            :substitutions:

            torchrun_local run_training.py \
                --task mug_in_drawer \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --demos_train 0-6 \
                --demos_valset 7-9 \
                --dataset <LOCAL_DATASET_PATH> \
                --base_log_dir <OUTPUT_DIR>

    .. tab:: Drill in Box

        .. code-block:: bash
            :substitutions:

            torchrun_local run_training.py \
                --task drill_in_box \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --demos_train 0-6 \
                --demos_valset 7-9 \
                --dataset <LOCAL_DATASET_PATH> \
                --base_log_dir <OUTPUT_DIR>

    .. tab:: Stick in Bin

        .. code-block:: bash
            :substitutions:

            torchrun_local run_training.py \
                --task stick_in_bin \
                --data_type rgbd_and_mesh \
                --feature_type radio_v25_b \
                --demos_train 0-6 \
                --demos_valset 7-9 \
                --dataset <LOCAL_DATASET_PATH> \
                --base_log_dir <OUTPUT_DIR>

Training Configuration
----------------------

- **Training demonstrations**: Demos 0-6
- **Validation demonstrations**: Demos 7-9
- **Data type**: RGBD and mesh features for comprehensive spatial understanding
- **Feature type**: Radio v2.5 B features for robust visual representation

Replace the following placeholders:

- ``<LOCAL_DATASET_PATH>``: Path to your dataset directory
- ``<OUTPUT_DIR>``: Directory where checkpoints and logs will be saved

.. note::
    The pre-trained checkpoints available in :doc:`download_models` were trained on 100+ demonstrations.
    If you want to train on more than the 10 demonstrations provided in :doc:`download_datasets`,
    you will need to :doc:`generate additional datasets <data_generation>` first.

.. include:: ../snippets/parameter_note_snippet.rst

.. _mindmap_checkpoint_structure:

Checkpoint Structure
--------------------

Training checkpoints are automatically saved in the following structure:

.. code-block:: text

   📂 <OUTPUT_DIR>/checkpoints/<DATE_TIME_OF_TRAINING_START>/
   ├── best.pth
   ├── last.pth
   └── training_args.json

Checkpoint Files
----------------

- ``best.pth``: The checkpoint with the lowest validation loss during training (recommended for evaluation)
- ``last.pth``: The checkpoint from the final training epoch
- ``training_args.json``: Complete model configuration and training parameters used

Model Configuration
-------------------

When running a checkpoint in :doc:`open loop evaluation <open_loop_evaluation>` or :doc:`closed loop evaluation <closed_loop_evaluation>`,
the model automatically loads its configuration from ``training_args.json``. This ensures consistency between training and evaluation.

To override the saved configuration, use the ``--ignore_model_args_json`` flag when running evaluation scripts.

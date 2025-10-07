Parameters
==========

All parameters are defined in :mindmap_code_link:`<mindmap/cli/args.py>`.
This page provides an overview of the most important parameters and their applicability to each workflow step.

Parameter Overview
------------------

The following table lists the key parameters and indicates which workflow steps they apply to:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
     - :doc:`Data Gen <data_generation>`
     - :doc:`Train <training>`
     - :doc:`Open Loop <open_loop_evaluation>`
     - :doc:`Closed Loop <closed_loop_evaluation>`
   * - ``task``
     - Task name (options: ``cube_stacking``, ``mug_in_drawer``, ``drill_in_box``, ``stick_in_bin``)
     - ✅
     - ✅
     - ✅
     - ✅
   * - ``data_type``
     - Data type (options: ``rgbd_and_mesh``, ``rgbd``, ``mesh``)
     - ✅
     - ✅
     - ✅
     - ✅
   * - ``feature_type``
     - Feature type (options: ``radio_v25_b``, ``clip_resnet50_fpn``, ``dino_v2_vits14``, ``rgb``)
     - ✅
     - ✅
     - ✅
     - ✅
   * - ``add_external_cam``
     - Whether to add an external camera as additional input to the model (and mapping)
     - ✅
     - ✅
     - ✅
     - ✅
   * - ``demos_datagen``
     - Demonstration range for data generation (supports ranges like ``0-9``)
     - ✅
     - ❌
     - ❌
     - ❌
   * - ``visualize``
     - Whether to visualize the data generation, training, or closed loop evaluation process (open loop evaluation is always visualized)
     - ✅
     - ✅
     - ❌
     - ✅
   * - ``output_dir``
     - Path to the output directory to create the dataset
     - ✅
     - ❌
     - ❌
     - ❌
   * - ``hdf5_file``
     - Path to the HDF5 file containing simulation environment and demonstration trajectories
     - ✅
     - ❌
     - ❌
     - ✅
   * - ``headless``
     - Whether to run the simulation in headless mode
     - ✅
     - ❌
     - ❌
     - ✅
   * - ``base_log_dir``
     - Path for storing Weights & Biases logs and checkpoints
     - ❌
     - ✅
     - ✅
     - ❌
   * - ``dataset``
     - Path to the dataset directory to load (in closed loop evaluation only needed in combination with ``--demo_mode execute_gt_goals``)
     - ❌
     - ✅
     - ✅
     - ✅
   * - ``demos_train``
     - Demonstration range of the training set (supports ranges like ``0-9``)
     - ❌
     - ✅
     - ❌
     - ❌
   * - ``demos_valset``
     - Demonstration range of the validation set during training (supports ranges like ``0-9``)
     - ❌
     - ✅
     - ❌
     - ❌
   * - ``train_iters``
     - Number of training iterations
     - ❌
     - ✅
     - ❌
     - ❌
   * - ``batch_size``
     - Batch size for the training set
     - ❌
     - ✅
     - ❌
     - ❌
   * - ``batch_size_val``
     - Batch size for the validation set
     - ❌
     - ✅
     - ❌
     - ❌
   * - ``demos_open_loop``
     - Demonstration range for open loop evaluation (supports ranges like ``0-9``)
     - ❌
     - ❌
     - ✅
     - ❌
   * - ``checkpoint``
     - Path to the `.pth` checkpoint file to load (optional in case of open loop evaluation and not needed in closed loop evaluation with ``--demo_mode execute_gt_goals``)
     - ❌
     - ❌
     - ✅
     - ✅
   * - ``--ignore_model_args_json``
     - Whether to ignore the model arguments JSON file (``training_args.json``). Per default, the model arguments are loaded from the file to ensure consistency between training and evaluation.
     - ❌
     - ❌
     - ✅
     - ✅
   * - ``demos_closed_loop``
     - Demonstration range for closed loop evaluation (supports ranges like ``0-9``)
     - ❌
     - ❌
     - ❌
     - ✅
   * - ``demo_mode``
     - How to run closed loop evaluation (options: ``execute_gt_goals`` for running GT goals from a dataset and ``closed_loop_wait`` for running inference once the last predicted goal is reached)
     - ❌
     - ❌
     - ❌
     - ✅
   * - ``record_videos``
     - Whether to record the closed loop evaluation runs (specify the output directory with ``--record_camera_output_path <VIDEO_OUTPUT_DIR>``)
     - ❌
     - ❌
     - ❌
     - ✅
   * - ``record_camera_output_path``
     - Path to the directory to store the recorded closed loop evaluation videos (only used in combination with ``--record_videos``)
     - ❌
     - ❌
     - ❌
     - ✅


.. note::

  If you set parameters differently from default, you need to ensure they stay compatible across workflow steps.
  For example, when creating a dataset with ``--data_type mesh`` and/or ``--feature_type rgb``,
  training on that dataset will only work if you also set these parameters for training.

.. note::

   For detailed parameter definitions and their complete applicability, refer to the :mindmap_code_link:`<mindmap/cli/args.py>` file.
   Each workflow step has its own parameter class (e.g., ``TrainingAppArgs`` for training) that inherits from parent classes.

Download Datasets
=================

We provide pre-generated datasets for all four :doc:`benchmark tasks <mindmap_tasks>`, hosted on Hugging Face.
Each dataset contains mindmap-formatted data ready for training, along with the original HDF5 demonstration files.

Available Datasets
------------------

- **Cube Stacking:** `nvidia/PhysicalAI-Robotics-mindmap-Franka-Cube-Stacking <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-mindmap-Franka-Cube-Stacking>`_
- **Mug in Drawer:** `nvidia/PhysicalAI-Robotics-mindmap-Franka-Mug-in-Drawer <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-mindmap-Franka-Mug-in-Drawer>`_
- **Drill in Box:** `nvidia/PhysicalAI-Robotics-mindmap-GR1-Drill-in-Box <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-mindmap-GR1-Drill-in-Box>`_
- **Stick in Bin:** `nvidia/PhysicalAI-Robotics-mindmap-GR1-Stick-in-Bin <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-mindmap-GR1-Stick-in-Bin>`_

Each dataset includes:

- **Original HDF5 file** from Isaac Lab Mimic for data generation
- **10 demonstrations** in mindmap format, ready for training

Prerequisites
-------------

Before downloading, ensure you have:

1. **Hugging Face account** registered
2. **Read-access token** ready for authentication

Download Instructions
---------------------

#. Install the Hugging Face Hub CLI:

   .. code-block:: bash

      pip install -U "huggingface_hub[cli]"

#. Authenticate with your token:

   .. code-block:: bash

      hf auth login

#. Download a dataset for your chosen task:

   .. tabs::
       .. tab:: Cube Stacking

           .. code-block:: bash
               :substitutions:

               hf download \
                  |cube_stacking_HF_dataset| \
                  --repo-type dataset \
                  --local-dir <LOCAL_DATASET_PATH>

       .. tab:: Mug in Drawer

           .. code-block:: bash
               :substitutions:

               hf download \
                  |mug_in_drawer_HF_dataset| \
                  --repo-type dataset \
                  --local-dir <LOCAL_DATASET_PATH>

       .. tab:: Drill in Box

           .. code-block:: bash
               :substitutions:

               hf download \
                  |drill_in_box_HF_dataset| \
                  --repo-type dataset \
                  --local-dir <LOCAL_DATASET_PATH>

       .. tab:: Stick in Bin

           .. code-block:: bash
               :substitutions:

               hf download \
                  |stick_in_bin_HF_dataset| \
                  --repo-type dataset \
                  --local-dir <LOCAL_DATASET_PATH>

#. Extract the demonstration files:

   .. code-block:: bash

      python mindmap/scripts/untar_demos.py \
          --demos_dir <LOCAL_DATASET_PATH> \
          --num_processes 10 \
          --remove_tar

Dataset Structure
-----------------

After extraction, your downloaded dataset will have the following structure:

.. code-block:: text

   📂 <LOCAL_DATASET_PATH>
   ├── 📂 demo_00000/
   ├── 📂 demo_00001/
   ├── 📂 ...
   ├── 📂 demo_00009/
   ├── <HDF5_FILE_NAME>.hdf5
   └── README.md

Each ``demo_XXXXX/`` folder contains the mindmap-formatted data for one demonstration, while the HDF5 file contains the original Isaac Lab Mimic data.

For detailed information about the dataset structure, see :ref:`mindmap_dataset_structure`.

.. note::

   Replace ``<LOCAL_DATASET_PATH>`` with your desired local directory path for storing the dataset.

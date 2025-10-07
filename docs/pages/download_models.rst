Download Checkpoints
====================

We provide pre-trained model checkpoints for all four benchmark tasks, hosted on Hugging Face.
These checkpoints allow you to immediately evaluate mindmap performance without training your own models.

Available Checkpoints
---------------------

- **mindmap Checkpoints:** `nvidia/PhysicalAI-Robotics-mindmap-Checkpoints <https://huggingface.co/nvidia/PhysicalAI-Robotics-mindmap-Checkpoints>`_

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

#. Download the checkpoints:

   .. code-block:: bash
       :substitutions:

       hf download \
           |mindmap_checkpoints_HF| \
           --local-dir <LOCAL_CHECKPOINT_PATH>

Checkpoint Structure
--------------------

The downloaded repository contains a folder for each task with the following structure:

.. code-block:: text

   📂 <LOCAL_CHECKPOINT_PATH>
   ├── 📂 cube_stacking_checkpoint
   │   ├── best.pth
   │   ├── last.pth
   │   └── training_args.json
   ├── 📂 mug_in_drawer_checkpoint
   │   ├── ...
   ├── 📂 drill_in_box_checkpoint
   │   ├── ...
   ├── 📂 stick_in_bin_checkpoint
   │   ├── ...
   └── README.md

For detailed information about the checkpoint structure, see :ref:`mindmap_checkpoint_structure`.

.. note::

   Replace ``<LOCAL_CHECKPOINT_PATH>`` with your desired local directory path for storing the checkpoints.

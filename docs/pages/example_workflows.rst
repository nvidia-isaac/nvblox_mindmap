Example Workflows
=================

This page provides an overview of the complete mindmap workflow and the available shortcuts to get you started quickly.

Workflow Overview
-----------------

.. figure:: ../images/mindmap_workflows.png
   :name: figure_mindmap_workflows

   mindmap Workflow Diagram

Complete Workflow
-----------------

To train and evaluate a mindmap model on your own custom task, follow these steps:

#. :doc:`Create a mindmap task <create_task>` - Define your custom robotic task
#. :doc:`Record demonstrations <demo_recording>` - Collect expert demonstrations for imitation learning
#. :doc:`Generate a mindmap dataset <data_generation>` - Process demonstrations into mindmap format
#. :doc:`Train a mindmap model <training>` - Train the spatial memory model
#. :doc:`Evaluate open loop <open_loop_evaluation>` - Test model predictions on recorded data
#. :doc:`Evaluate closed loop <closed_loop_evaluation>` - Test model performance in Isaac Lab simulation

Quick Start Options
-------------------

We provide several shortcuts to help you get started immediately:

**Predefined Tasks:**

Four benchmark tasks (``Cube Stacking``, ``Mug in Drawer``, ``Drill in Box``, ``Stick in Bin``) with complete data and models ready to use.

**Downloadable Resources:**

- HDF5 demonstration files for all tasks
- Pre-generated datasets in mindmap format
- Trained model checkpoints ready for evaluation

**These resources allow you to:**

- Skip directly to any workflow step
- Use them as references when creating your own tasks
- Quickly test mindmap capabilities without going through the complete workflow

Getting Started
---------------

See the :ref:`Workflow Diagram <figure_mindmap_workflows>` above for a visual overview of the complete workflow and available shortcuts.

**For detailed information about:**

- Predefined tasks: :doc:`mindmap_tasks`
- Pre-generated datasets including HDF5 demonstration files: :doc:`download_datasets`
- Trained model checkpoints: :doc:`download_models`

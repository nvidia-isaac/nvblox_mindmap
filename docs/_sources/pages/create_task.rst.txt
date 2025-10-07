Creating New Tasks
==================

This guide explains how to create and integrate new tasks into the mindmap system. Tasks in mindmap are built on top of Isaac Lab's manager-based RL environment framework and follow a structured approach for defining robotic manipulation scenarios.

For comprehensive background on Isaac Lab concepts, refer to the `Isaac Lab Documentation <https://isaac-sim.github.io/IsaacLab/main/index.html>`_.

Overview
--------

A task in mindmap represents a complete robotic manipulation scenario that includes:

- **Scene Configuration**: Defines the 3D environment, including robots, objects, cameras, and other assets
- **Task Logic**: Implements the core task behavior through observations, actions, and events
- **Success Criteria**: Determines when a task is completed successfully
- **Environment Registration**: Makes the task available to the training and evaluation pipeline

The mindmap system currently supports four benchmark manipulation tasks that require spatial memory capabilities: **Cube Stacking**, **Mug in Drawer**, **Drill in Box**, and **Stick in Bin**. These tasks are specifically designed to evaluate spatial awareness beyond single camera views. For detailed descriptions of each task and their spatial memory requirements, see :doc:`mindmap_tasks`.

Task Structure
--------------

Each task is organized in the following directory structure under :mindmap_code_link:`<mindmap/tasks/task_definitions>`

.. code-block::

    your_task_name/
    ├── __init__.py
    ├── your_task_env_cfg.py          # Base scene configuration
    ├── config/                       # Robot-specific configurations
    │   ├── __init__.py
    │   └── robot_name/               # e.g., franka, gr1
    │       ├── __init__.py           # Environment registration
    │       ├── your_task_robot_env_cfg.py
    └── mdp/                          # Task-specific logic
        ├── __init__.py
        ├── observations.py           # Custom observation functions
        ├── terminations.py           # Success criteria and termination conditions
        └── events.py                 # Scene randomization and initialization

Key Components
--------------

Scene Configuration
~~~~~~~~~~~~~~~~~~~

The base scene configuration (``your_task_env_cfg.py``) defines the 3D environment using Isaac Lab's scene system:
For detailed guidance on scene design, see the `Creating a Manager-Based Base Environment <https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html>`_ in the Isaac Lab documentation.

Success Criteria
~~~~~~~~~~~~~~~~~

The ``mdp/terminations.py`` file defines when a task is considered complete. Success criteria typically check:

- **Object positioning**: Whether objects reach target locations within tolerance
- **Robot state**: Gripper position, joint configurations

For example, the Mug in Drawer task :mindmap_code_link:`<mindmap/tasks/task_definitions/mug_in_drawer/mdp/terminations.py>` checks that:

1. The mug is positioned within the target drawer bounds (x, y, z coordinates)
2. The gripper is in an open position (indicating the mug was released)
3. Both conditions are satisfied simultaneously

For detailed guidance on implementing termination criteria, see `Creating a Manager-Based RL Environment <https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html>`_.


Task Registration
~~~~~~~~~~~~~~~~~

Tasks are registered with the Gymnasium environment registry in the robot-specific ``__init__.py`` files:

.. code-block:: python

    gym.register(
        id="Isaac-Your-Task-Robot-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": your_task_robot_env_cfg.YourTaskRobotEnvCfg,
        },
        disable_env_checker=True,
    )

For comprehensive information on environment registration, see `Registering an Environment <https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html>`_ in the Isaac Lab documentation.


Mimic Environments for Teleoperation and Data Generation
--------------------------------------------------------

mindmap includes specialized **Mimic environments** that extend the standard task environments to support demonstration data augmentation. These environments are built on Isaac Lab's Mimic framework and take existing recorded demonstrations to automatically generate additional synthetic demonstrations, multiplying the available training data.

For comprehensive information on teleoperation, data generation workflows, and Isaac Lab Mimic capabilities, see the `Teleoperation and Imitation Learning with Isaac Lab Mimic <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>`_ documentation.

Available Mimic Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following mimic environments are currently available:

- **Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0**: Franka arm cube stacking
- **Isaac-Mug-in-Drawer-Franka-Mimic-v0**: Franka arm placing mug in drawer
- **Isaac-Drill-In-Box-GR1T2-Left/Right-Mimic-v0**: GR1 robot drill insertion (left/right variants)
- **Isaac-Stick-In-Bin-GR1T2-Left/Right-Mimic-v0**: GR1 robot stick placement (left/right variants)

.. note::

  GR1 tasks require separate left and right hand variants because Isaac Lab Mimic requires specifying subtasks for specific hands.
  Demonstrations are recorded for each hand separately and later combined into a single dataset.

Creating Mimic Environments for New Tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mimic environments follow a specific structure within the mindmap codebase.
Each mimic environment is organized in :mindmap_code_link:`<mindmap/tasks/task_definitions/mimic_envs>` with the following structure:

.. code-block::

    mimic_envs/
    ├── __init__.py                           # Environment registrations
    ├── your_task_robot_mimic_env.py          # Mimic environment implementation
    ├── your_task_robot_mimic_env_cfg.py
    └── ...

**Environment Implementation** (``your_task_robot_mimic_env.py``):
  - Inherits from ``ManagerBasedRLMimicEnv``

**Configuration** (``your_task_robot_mimic_env_cfg.py``):
  - Inherits from both your task's environment config and ``MimicEnvCfg``
  - Defines ``subtask_configs`` with ``SubTaskConfig`` objects for each task phase
  - Configures data generation parameters (noise, interpolation, selection strategies)

**Registration** (``__init__.py``):
  - Registers the mimic environment with gymnasium using the mimic entry point
  - Example: ``Isaac-Your-Task-Robot-Mimic-v0``

For detailed implementation guidance, refer to the `Creating Your Own Isaac Lab Mimic Compatible Environments <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#creating-your-own-isaac-lab-mimic-compatible-environments>`_ section in the IsaacLab documentation.

Integration with mindmap
------------------------

To integrate your task with the mindmap pipeline:

1. **Add to Tasks Enum**: Update :mindmap_code_link:`<mindmap/tasks/tasks.py>` to include your task in the ``Tasks`` enum.
2. **Add Task Name Mapping**: Add the full Isaac Lab task name in :mindmap_code_link:`<mindmap/tasks/tasks.py>`.
3. **Add Success Function**: Update :mindmap_code_link:`<mindmap/tasks/task_success.py>` to include your task's success criteria.

Example: Mug-in-Drawer Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Mug in Drawer** task demonstrates a complete task implementation:

- **Objective**: Place a mug inside a target kitchen drawer marked with other mugs
- **Scene**: Kitchen environment with a table, drawer, and target mug
- **Success Criteria**: Mug positioned within drawer bounds AND gripper open
- **Robots Supported**: Franka Panda arm

The success function checks spatial bounds relative to the drawer bottom and ensures the gripper has released the object.

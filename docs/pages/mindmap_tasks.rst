mindmap Tasks
=============

mindmap includes four benchmark tasks designed to evaluate spatial memory capabilities in robotic manipulation.
These tasks are specifically chosen because they require the robot to maintain spatial awareness beyond what can be seen in a single camera view.

Task Overview
-------------

.. list-table::
    :class: gallery

    * - .. figure:: ../images/cube_stacking_task.jpg
         :width: 300px

         Cube Stacking
      - .. figure:: ../images/mug_in_drawer_task.jpg
         :width: 300px

         Mug in Drawer
    * - .. figure:: ../images/drill_in_box_task.jpg
         :width: 300px

         Drill in Box
      - .. figure:: ../images/stick_in_bin_task.jpg
         :width: 300px

         Stick in Bin

The four tasks, shown above from top-left in clockwise order, are:

* **Cube Stacking**: Stack three cubes in a specific order (blue bottom, red middle, green top). Initial cube positions are randomized, requiring the robot to remember cube locations.

* **Mug in Drawer**: Place a mug into a drawer containing other mugs. Object positions on the kitchen counter are randomized, and the destination drawer position is permuted, making spatial memory essential.

* **Drill in Box**: Place a hand drill into an open box. The drill position is randomized, and open and closed boxes are permuted, requiring the robot to remember the open box location.

* **Stick in Bin**: Place a candlestick into a bin. Both the stick and bin positions are randomized, necessitating spatial memory to locate and navigate to the target.

Spatial Memory Challenge
------------------------

All tasks are designed with a critical constraint: the robot receives only a single egocentric camera view that cannot capture the entire task space within its field of view.
This limitation means that successful task completion requires the robot to:

1. **Remember object locations** that are no longer visible
2. **Navigate to remembered locations** based on spatial memory
3. **Maintain spatial awareness** throughout the task execution

Without spatial memory, policies would fail to reach high success rates because they cannot see the complete task environment at any given moment.

Feature Reconstruction
----------------------

mindmap addresses this challenge by providing feature reconstruction as input to the policy, enabling the model to maintain spatial memory:

.. list-table::
    :class: gallery

    * - .. figure:: ../images/cube_stacking_reconstruction.jpg
         :width: 300px

         Cube Stacking
      - .. figure:: ../images/mug_in_drawer_reconstruction.jpg
         :width: 300px

         Mug in Drawer
    * - .. figure:: ../images/drill_in_box_reconstruction.jpg
         :width: 300px

         Drill in Box
      - .. figure:: ../images/stick_in_bin_reconstruction.jpg
         :width: 300px

         Stick in Bin

The reconstruction provides a 3D spatial representation that allows the policy to "remember" where objects are located, even when they are outside the current camera view.

For more details about why spatial memory is crucial for these and many other realistic robotic tasks, refer to our :ref:`mindmap paper <Papers>`.

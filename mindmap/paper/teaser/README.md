# Paper - Teaser Render Instructions

**Author:** Alex Millane
**Date:** 20205.08.08


The goal here is to produce a sexy render for the front page of the paper.
The goal with the document is to describe how to recreate the render I put together.

## Prerequisites

You need to install on top of the `mindmap_deps` docker image:

```
pip install usd-core
```

## Step 1 - Produce the nvblox maps

We produce the nvblox maps for the render by saving a full nvblox map as each time step
of a datagen run.

```
python run_isaaclab_datagen.py --task drill_in_box --hdf5_file /datasets/mimicgen/galileo_gr1_left_10_demos_v4.hdf5 --output_dir datagen_output --save_serialized_nvblox_map_to_disk
```

We now convert the maps to USD

```
python3 /workspaces/mindmap/mindmap/paper/teaser/convert_maps_usd.py --input_dir datagen_output/demo_00000/ --visualize
```

> Note that this will take awhile.


## Record the robot motion

In order to record a motion we need to run without fabric. See [here](https://isaac-sim.github.io/IsaacLab/main/source/how-to/record_animation.html)

So we run a datagen, which provides GT motions, with fabric disabled and record the motion.

```
python run_isaaclab_datagen.py --task drill_in_box --hdf5_file /datasets/mimicgen/galileo_gr1_left_10_demos_v4.hdf5 --output_dir datagen_output --disable_fabric
```


## Open Isaac Sim, Load Scene, Render


Open Isaac Sim with the modified experience file:

```
/workspaces/mindmap/mindmap/paper/teaser/start_isaac_sim_for_rendering.sh
```

Load your two recorded motion files by right clicking on them and selecting "Insert as sublayer".

Add in the reconstruction from the UI by navigating to it and selecting "Insert as sublayer".

Press the play button to get the robot to a position that looks good.

Now to render:
- click the "Syntetic Data Recorder" tab in the lower right
- Click the render settings in the upper right
- Select "interactive (path tracing)"
- Set up the number of "samples to pixels to 20"
- Change "RTSubframes" to 2 (in the lower right)
- Set the output directory
- Change the viewport renderer in the upper left to "RTX - Interactive"
- Click the start button in the lower right.

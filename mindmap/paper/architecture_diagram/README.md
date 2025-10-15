# Architecture Diagrams

Architecture Diagram: [link](https://drive.google.com/file/d/1Zhz6qLruwS4h6vrYDl2fNNewfpOuIPxb/view?usp=drive_link)

This script notebook in this folder generates a number of thumbnail images that form part of the architecture diagram.

The notebook requires as input, the output of isaaclab datagen. I.e. the output of the command:

```
python mindmap/run_isaaclab_datagen.py --task cube_stacking --hdf5_file /datasets/mimicgen/1000_demos_rgbd_caminfo_new.hdf5 --output_dir datagen_output/ --save_serialized_nvblox_map --output_data_type rgbd_and_mesh --wrist_cam_only
```

Then run script on the output directory to generate the figures

```
python paper/architecture_diagram/architecture_diagram.py --input_dir datagen_output/demo_00000
```

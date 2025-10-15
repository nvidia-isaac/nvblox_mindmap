# Reconstruction figures

This folder contains scripts for generating the reconstruction figures.

## Prerequisites

We have some additional dependencies over what the docker image comes with:

```
sudo apt-get install imagemagick
```

## Input data

I have stored the input reconstructions, PCA parameters and viewpoints on OSMO. To get started download the input data

```
./download_data.sh
```

This will automatically download and extract the data to the right location.

> Note that if you decide to change the data, just replace the files, and then you can update the dataset on OSMO by running `download_data.sh -u` which will reupload a new version.

## Generate figures

The script is called once per task.

```
paper/reconstructions/generate_reconstruction_figures.py --task_name cube_stacking
paper/reconstructions/generate_reconstruction_figures.py --task_name mug_in_drawer
paper/reconstructions/generate_reconstruction_figures.py --task_name drill_in_box
paper/reconstructions/generate_reconstruction_figures.py --task_name stick_in_bin
```

> Note: Don't change the viewpoint in the displayed image, because it will save with the wrong viewpoint.

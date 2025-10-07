# mindmap: Spatial Memory in Deep Feature Maps for 3D Action Policies

[![Isaac Sim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-maroon.svg)](https://isaac-sim.github.io/IsaacLab/v2.1.0/index.html)
[![nvblox](https://img.shields.io/badge/nvblox-public-darkgreen.svg)](https://github.com/nvidia-isaac/nvblox/tree/public)
[![pytorch](https://img.shields.io/badge/pytorch-2.4-darkorange.svg)](https://docs.pytorch.org/docs/2.4/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3.10/)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-purple.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-NVIDIA-yellow.svg)](LICENSE.md)

<table>
  <tr>
    <td align="center">
      <img src="docs/images/teaser_isaaclab.jpg" height="400px"/><br/>
      Humanoid robot performing the <code>Drill in Box</code> spatial memory task in IsaacLab simulation.
    </td>
    <td align="center">
      <img src="docs/images/teaser_reconstruction.png" height="400px"/><br/>
      Deep feature map (colored by PCA) that enables spatial memory for the <code>mindmap</code> 3D action policy.
    </td>
  </tr>
</table>

## What is mindmap?

mindmap is a 3D diffusion policy that generates robot trajectories based on semantic 3D reconstruction of the environment.
Unlike traditional approaches that rely solely on current visual input,
mindmap maintains spatial memory through deep feature maps,
enabling robots to perform complex manipulation tasks that require remembering the layout of the environment and the locations of objects.

Our [mindmap paper](#papers) demonstrates that mindmap effectively solves manipulation
tasks where state-of-the-art approaches without memory mechanisms struggle. Through extensive simulation experiments, we show significant improvements in task success rates for scenarios requiring spatial memory.

mindmap builds upon the [nvblox codebase](https://github.com/nvidia-isaac/nvblox/tree/public),
which provides a GPU-accelerated reconstruction system wrapped in PyTorch. This system is essential for building the deep feature maps that enable spatial memory in mindmap.


## Open Source Release


We release our complete codebase including:

- **Reconstruction System**: Real-time 3D mapping and feature extraction
- **Training Code**: End-to-end model training pipeline
- **Evaluation Tasks**: Four benchmark tasks for spatial memory testing
- **Pre-trained Models**: Ready-to-use [checkpoints](https://huggingface.co/nvidia/PhysicalAI-Robotics-mindmap-Checkpoints) for immediate evaluation


## Getting Started

To get started with `mindmap`, see our [documentation site](https://nvlabs.github.io/nvblox_mindmap/).

## License
This project is licensed under the **NVIDIA License (NSCLv1)**.
See [LICENSE.md](LICENSE.md) for details.


## Papers

If you find mindmap useful for your research, please consider citing our work:

* Remo Steiner, Alexander Millane, David Tingdahl, Clemens Volk, Vikram Ramasamy, Xinjie Yao, Peter Du, Soha Pouya and Shiwei Sheng.
  "**mindmap: Spatial Memory in Deep Feature Maps for 3D Action Policies**". CoRL 2025 Workshop RemembeRL.
  [arXiv preprint arXiv:2509.20297 (2025).](https://arxiv.org/abs/2509.20297)

* Alexander Millane, Helen Oleynikova, Emilie Wirbel, Remo Steiner, Vikram Ramasamy, David Tingdahl, and Roland Siegwart.
  **nvblox: GPU-Accelerated Incremental Signed Distance Field Mapping**.  ICRA 2024.
  [arXiv preprint arXiv:2311.00626 (2024).](https://arxiv.org/abs/2311.00626)

# Model Overview

### Description:

``mindmap`` is a 3D diffusion policy that generates robot trajectories based on a semantic 3D reconstruction of the environment, enabling robots with spatial memory.

Trained models are available on Hugging Face: [PhysicalAI-Robotics-mindmap-Checkpoints](https://huggingface.co/nvidia/PhysicalAI-Robotics-mindmap-Checkpoints)

This model is ready for commercial/non-commercial use

### License/Terms of Use

- Model: [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- Code: [NVIDIA License (NSCLv1)](https://github.com/NVlabs/nvblox_mindmap/tree/public/LICENSE.md)

### Deployment Geography:

Global

### Use Case

The trained ``mindmap`` policies allow for quick evaluation of the ``mindmap`` concept on selected simulated robotic manipulation tasks.

- Researchers, Academics, Open-Source Community: AI-driven robotics research and algorithm development.
- Developers: Integrate and customize AI for various robotic applications.
- Startups & Companies: Accelerate robotics development and reduce training costs.


### Release Date

Github 09/26/2025 via github.com/NVlabs/nvblox_mindmap
Hugging Face 09/26/2025 via huggingface.co/nvidia/PhysicalAI-Robotics-mindmap-Checkpoints

## References(s):

- ``mindmap`` paper:

    - Remo Steiner, Alexander Millane, David Tingdahl, Clemens Volk, Vikram Ramasamy, Xinjie Yao, Peter Du, Soha Pouya and Shiwei Sheng. "**mindmap: Spatial Memory in Deep Feature Maps for 3D Action Policies**". CoRL 2025 Workshop RemembeRL.
  [arXiv preprint arXiv:2509.20297 (2025).](https://arxiv.org/abs/2509.20297)
- ``mindmap`` codebase:
    - github.com/NVlabs/nvblox_mindmap

## Model Architecture:

**Architecture Type:** Denoising Diffusion Probabilistic Model


**Network Architecture:**

``mindmap`` is a Denoising Diffusion Probabilistic Model that samples robot trajectories conditioned on sensor observations and a 3D reconstruction of the environment. Images are first passed through a Vision Foundation Model and then back-projected, using the depth image, to a pointcloud. In parallel, a reconstruction of the scene is built that accumulates metric-semantic information from past observations. The two 3D data sources, the instantaneous visual observation and the reconstruction, are passed to a transformer that iteratively denoises robot trajectories.

**This model was developed based on:** [3D Diffuser Actor](https://3d-diffuser-actor.github.io/)

**Number of model parameters:** ∼3M trainable, plus ∼100M frozen in the image encoder

## Input:

**Input Type(s):**
- RGB: Image frames
- Geometry: Depth frames converted to 3D pointclouds
- State: Robot proprioception
- Reconstruction: Metric-semantic reconstruction represented as featurized pointcloud

**Input Format(s):**
- RGB: float32 in the range `[0, 1]`
- Geometry: float32 in world coordinates
- State: float32 in world coordinates
- Reconstruction (represented as feature pointcloud):
    - Points: float32 in world coordinates
    - Features: float32

**Input Parameters:**
- RGB: `[NUM_CAMERAS, 3, HEIGHT, WIDTH]` - 512x512 resolution on the provided checkpoints
- Geometry: `[NUM_CAMERAS, 3, HEIGHT, WIDTH]` - 512x512 resolution on the provided checkpoints
- State: `[HISTORY_LENGTH, NUM_GRIPPERS, 8]` - consisting of end-effector translation, rotation (quaternion, wxyz) and closedness
- Reconstruction (represented as feature pointcloud):
    - Points: `[NUM_POINTS, 3]` - `NUM_POINTS` is 2048 for the provided checkpoints
    - Features: `[NUM_POINTS, FEATURE_DIM]` - `FEATURE_DIM` is 768 for the `RADIO_V25_B` feature extractor used for the provided checkpoints

## Output:

**Output Type(s):** Robot actions

**Output Format:** float32

**Output Parameters:**
- Gripper: `[PREDICTION_HORIZON, NUM_GRIPPERS, 8]` - consisting of end-effector translation, rotation (quaternion, wxyz) and closedness
- Head Yaw: `[PREDICTION_HORIZON, 1]` - only for humanoid embodiments

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):** PyTorch

**Supported Hardware Microarchitecture Compatibility:**
- NVIDIA Ampere
- NVIDIA Blackwell
- NVIDIA Jetson
- NVIDIA Hopper
- NVIDIA Lovelace
- NVIDIA Pascal
- NVIDIA Turing
- NVIDIA Volta


**Preferred/Supported Operating System(s):**
* Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):

This is the initial version of the model, version 1.0.0

## Training, Testing, and Evaluation Datasets:

Datasets:
- cube_stacking_checkpoint: [Franka Cube Stacking Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-mindmap-Franka-Cube-Stacking)
- mug_in_drawer_checkpoint: [Franka Mug in Drawer Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-mindmap-Franka-Mug-in-Drawer)
- drill_in_box_checkpoint: [GR1 Drill in Box Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-mindmap-GR1-Drill-in-Box)
- stick_in_bin_checkpoint: [GR1 Stick in Bin Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-mindmap-GR1-Stick-in-Bin)

**Data Modality:** Image, 3D reconstruction, robot states

**Image Training Data Size:** Less than a Million Images

**3D reconstruction, robot state Data Size:** Less than a Million Samples

**Data Collection Method by dataset:**
* Synthetic
* Human teleoperation
* Automatic trajectory generation

**Properties:**

The models were trained on 100 (GR1) and 130 (Franka) demonstrations. The evaluation set consisted of 20 distinct demonstrations. Closed loop testing was performed on 100 demonstrations mutually exclusive from the training set. The training data is synthetic only and fully generated in Isaac Lab.

# Inference:

**Engine:** PyTorch

**Test Hardware:** Linux, L40S

## Model Limitations:

This model is not tested or intended for use in mission critical applications that require functional safety. The use of the model in those applications is at the user's own risk and sole responsibility, including taking the necessary steps to add needed guardrails or safety mechanisms.

- Limitation: This policy is only effective in the exact simulation environment in which it was trained.
    - Mitigation: Recommended to retrain the model in new simulation environments.
- Limitation: The policy was not tested on a physical robot and likely only works in simulation.
    - Mitigation: Expand training, testing and validation on physical robot platforms.

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

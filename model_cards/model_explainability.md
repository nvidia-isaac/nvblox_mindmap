Field                                                                                                  |  Response
:------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------
Intended Task/Domain:                                                                   |  Robotic Manipulation
Model Type:                                                                                            |  Denoising Diffusion Probabilistic Model
Intended Users:                                                                                        |  Roboticists and researchers in academia and industry who are interested in robot manipulation research
Output:                                                                                                |  Actions consisting of end-effector poses, gripper states and head orientation.
Describe how the model works:                                                                          |  ``mindmap`` is a Denoising Diffusion Probabilistic Model that samples robot trajectories conditioned on sensor observations and a 3D reconstruction of the environment.
Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of:  |  Not Applicable
Technical Limitations & Mitigation:                                                                    |  - Limitation: This policy is only effective in the exact simulation environment in which it was trained. Mitigation: Recommended to retrain the model in new simulation environments. - Limitation: The policy was not tested on a physical robot and likely only works in simulation. Mitigation: Expand training, testing and validation on physical robot platforms.
Verified to have met prescribed NVIDIA quality standards:  |  Yes
Performance Metrics:                                                                                   |  Closed loop success rate on simulated robotic manipulation tasks.
Potential Known Risks:                                                                                 |  The model might be susceptible to rendering changes on the simulation tasks it was trained on.
Licensing:                                                                                             |  [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

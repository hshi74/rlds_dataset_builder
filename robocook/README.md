This is the real-world robot manipulation dataset collected for [RoboCook: Long-Horizon Elasto-Plastic Object Manipulation with Diverse Tools](https://hshi74.github.io/robocook/).

We have collected approximately 20 minutes of interaction data between each tool and the dough using a policy that randomly samples actions from a predefined action space. After each episode, a human reshapes the dough into a different random configuration.

The data breakdown per tool is as follows:

1. Asymmetric gripper / two-rod symmetric gripper / two-plane symmetric gripper: 300 episodes each; 
1. Circle press / square press / circle punch / square punch: 270 episodes each;
1. Large roller / small roller: 240 episodes each.

In total, this aggregates to 2460 episodes, with each episode comprising roughly 50 frames.

We use the 7-DoF Franka Emika Panda robot arm and its parallel jaw gripper as the base robot. Four calibrated Intel RealSense D415 RGB-D cameras are fixed on vertical metal bars around the robot tabletop. The cameras capture **1280×720** RGB-D images. We also design a set of 3D-printed tools based on real-world dough manipulation tools. Please refer to the [paper](https://arxiv.org/abs/2306.14447) for more details on our setup.

During the data collection phase, we captured a temporally and spatially smoothed point cloud stream. To optimize storage and maintain compatibility with the RLDS dataset format, we projected the point cloud back to an RGB-D image. Consequently, it's typical to notice gaps in the image, indicative of filtered-out noisy depth values. We also include both camera extrinsics and intrinsics within the dataset, facilitating point cloud reconstruction if needed.

A visualization of the data collection process:

[![RoboCook Data Collection](https://img.youtube.com/vi/VxkOF6mS90I/0.jpg)](https://www.youtube.com/watch?v=VxkOF6mS90I)

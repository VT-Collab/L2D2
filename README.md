# L2D2: Robot Learning from 2D Drawings

This repository provides our implementation of L2D2 in a real world environment with a 6-DoF UR-10 robot arm. The videos for our experiments can be found on [YouTube](https://www.youtube.com/watch?v=ikLjj3e1w1g).

## Installation and Setup

To install L2D2 and the related submodules, clone this repo using
```
git clone git@github.com:VT-Collab/L2D2.git --recurse-submodules
```

Install the required packages and dependencies for running L2D2 on your machine.

- Conda Environment (recommended approach)
```
conda env create -f requirements.yml
```

- PyPI (pip)
```
pip install requirements.txt
```
Note that this method (pip) may not install all the necessary packages and that some packages may need manual installation.

### Setting up the object detection pipeline
We use the [Detic](https://github.com/facebookresearch/Detic/tree/main) pipeline for object detection in our framework. For any issues with the installation of the package please check the linked repository.

Navigate to `L2D2/DETIC/detectron2/` and run
```
python -m pip install -e detectron2
```
or 
```
pip install -e .
```

To install the object detection models, navigate to `L2D2/DETIC/Detic/` and run the following commands
```
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

This process should install the necessary packages and models for running L2D2 base code. However, you may need additional packages for running the scripts with your robotic system. 

For our implementation of `robot/`, we use [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu) for controlling the UR-10 robot arm. We use the following drivers for controlling the robot: [Universal_Robots_ROS_Driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver).

## Implementation
We break down the implementation of different parts of our approach in different instruction files:

- [Learning a Reconstruction Model](https://github.com/VT-Collab/L2D2/tree/main/docs/reconstruction.md): This file provides implementation details for data collection and training of the reconstruction model

- [Learning from Drawings](https://github.com/VT-Collab/L2D2/tree/main/docs/learning_from_drawings.md): This file provides instructions for collecting drawings and converting them to 3d robot demos, training an initial robot policy and evaluation of this policy.

- [Fine-Tuning using Physical Demonstrations](https://github.com/VT-Collab/L2D2/tree/main/docs/fine_tuning.md): This file outlines the process to collect physcial demonstrations on the robot and using them to update the reconstruction model and fine-tune the robot policy.

## Citation
If you find this project useful for your research, please use the following BibTeX entry:
```
@article{mehta2025l2d2,
  title={L2D2: Robot Learning from 2D Drawings},
  author={Mehta, Shaunak A and Nemlekar, Heramb and Sumant, Hari and Losey, Dylan P},
  journal={arXiv preprint arXiv:2505.12072},
  year={2025}
}
```
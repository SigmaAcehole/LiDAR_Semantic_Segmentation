# LiDAR Semantic Segmentation
## PointNet2
Pre-requisites:

## Tested environment and hardware
1. OS: Ubuntu 22.04 LTS (Used WSL2 but dual-boot should work as well)
2. GPU: Nvidia RTX 3050 Mobile (4 GB)
3. CPU: ARM Ryzen 5 5600H
4. ROS2 humble

## Setup environment
Recommended to setup a virtual environment using conda to avoid dependency issues with libraries. Code tested with Python 3.11.10, PyTorch 2.5.1 and CUDA 11.8.
1. Install Miniconda using intructions from [here](https://docs.anaconda.com/miniconda/#quick-command-line-install).
2. Create a conda environment with Python 3.11.10
```shell
conda create -n pointnet2 python=3.11.10
```
3. Activate the environment
```shell
conda activate pointnet2
```
4. Install PyTorch and tqdm    
```shell 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install tqdm
conda install pip=24.2
```
If any other library needs to be installed (ex: numpy), install it using conda. If library is not available in conda then use pip.


## Data preparation
1. Download Stanford 3D Indoor Scene (S3DIS) dataset.
2. Unzip.
3. Run this script.

## Training 
Pre-trained model is available `./log/pointnet2_sem_seg/checkpoints/best_model.pth`. However, if model needs to be trained on S3DIS dataset then follow the following instructions.
1. 

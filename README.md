# LiDAR Semantic Segmentation
## PointNet2
Pre-requisites:

## Tested environment
1. OS: Ubuntu 22.04 LTS (Used WSL2 but dual-boot should work as well)
2. Python 3.11.10
3. PyTorch 2.5.1 with CUDA 11.8.

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
1. Create the following directory to store dataset and processed dataset.
```shell
mkdir data/
mkdir data/stanford_indoor3d_downsampled
```
2. Download `Stanford3dDataset_v1.2_Aligned_Version.zip` from [here](https://cvg-data.inf.ethz.ch/s3dis/).
3. Unzip and store it inside `data/`. The directory structure should be `data/Stanford3dDataset_v1.2_Aligned_Version` now.
3. Run the dataset processing script. This script will collect each room from the original dataset, append label index, downsample it, format it as XYZRGBL and store it as `.npy` files. The processed `.npy` files will be stored inside `data/stanford_indoor3d_downsampled`.
```shell
python collect_downsample.py 
```
Both the train and test scripts use the created `.npy` files.

## Training 
Pre-trained model is available `./log/pointnet2_sem_seg/checkpoints/best_model.pth`. However, if model needs to be trained on S3DIS dataset then follow the following instructions.
1. 

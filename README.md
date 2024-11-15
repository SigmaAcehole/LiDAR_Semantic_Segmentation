# LiDAR Semantic Segmentation
## PointNet2
Pre-requisites:

## Tested environment
1. OS: Ubuntu 22.04 LTS (Used WSL2 but dual-boot should work as well)
2. Python 3.11.10
3. PyTorch 2.5.1 with CUDA 11.8.
4. Open3D 0.18.0

## Setup environment
Recommended to setup a virtual environment using conda to avoid dependency issues with libraries. Code tested with Python 3.11.10, PyTorch 2.5.1 and CUDA 11.8.
1. Install Miniconda.
```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
If there is any trouble installing miniconda, refer instructions from [here](https://docs.anaconda.com/miniconda/#quick-command-line-install).    

2. Refresh the terminal.
```shell
source ~/miniconda3/bin/activate
```
3. Initialize conda on all terminals.
```shell
conda init --all
```

4. Create a conda environment with Python 3.11.10
```shell
conda create -n pointnet2 python=3.11.10
```
5. Activate the environment.
```shell
conda activate pointnet2
```
6. Install all required packages.   
```shell 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install tqdm
conda install conda-forge::open3d
```
If any other package needs to be installed (ex: numpy), install it using conda. If package is not available in conda then use pip.


## Data preparation for Stanford 3D Indoor Scenes dataset
1. Make sure you are in the root directory of PointNet2 i.e., `LiDAR_Semantic_Segmentation/PointNet2`. Create the following directory to store original dataset and processed dataset. 
```shell
mkdir data/
mkdir data/stanford_indoor3d_downsampled
```
2. Download `Stanford3dDataset_v1.2_Aligned_Version.zip` from [here](https://cvg-data.inf.ethz.ch/s3dis/).
3. Unzip and store it inside `data/`. The directory structure should be `data/Stanford3dDataset_v1.2_Aligned_Version` now. If the original dataset is unzipped in a different location (ex: `/mnt/dataset` in the AI server of SMART lab), then change the `DATA_PATH` variable in line 13 of `data_utils/collect_downsample.py`.
3. Run the dataset processing script. This script will collect each room from the original dataset, append label index, downsample it, format it as XYZRGBL and store it as `.npy` files. The processed `.npy` files will be stored inside `data/stanford_indoor3d_downsampled`.
```shell
cd data_utils
python collect_downsample.py 
```
This step need not be repeated again as long as the `.npy` are stored in the correct folder. Both the test and train scripts use these `.npy` files as input. Currently this script randomly samples 50% of the original pointcloud to downsample it. If you want to change it then change the variable `sampling_ratio` in line 95 of `data_utils/collect_downsample.py`.

## Training 
Pre-trained model is available `log/sem_seg/pointnet2_sem_seg/checkpoints/best_model.pth`. However, if model needs to be trained on S3DIS dataset then run the train script. Make sure you have followed the data preparation steps in the previous section. From root directory of PointNet2 i.e., `LiDAR_Semantic_Segmentation/PointNet2`, run the train script.

```shell
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --batch_size 32 --epoch 32
```
It is recommended to use a smaller batch size if training on laptop. If the train script stops running with a "Killed" message then most likely it means it ran out of RAM memory, so reduce batch size in this case.

## Testing
Trained models are available inside `log/semseg`. 
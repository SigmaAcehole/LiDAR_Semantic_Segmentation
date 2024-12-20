# PointNet++
The code is an implementation of [PointNet++]{https://github.com/yanx27/Pointnet_Pointnet2_pytorch}.
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
conda install conda-forge::matplotlib
```
If any other package needs to be installed (ex: numpy), install it using conda. If package is not available in conda then use pip.


## Data preparation for Stanford 3D Indoor Scenes dataset
1. Make sure you are in the root directory of PointNet2 i.e., `LiDAR_Semantic_Segmentation/PointNet2`. Create the following directory to store original dataset and processed dataset. 
```shell
mkdir data/
mkdir data/stanford_indoor3d_downsampled
```
2. Download `Stanford3dDataset_v1.2_Aligned_Version.zip` from [here](https://cvg-data.inf.ethz.ch/s3dis/).
3. Unzip and store it inside `data/`. The directory structure should be `data/Stanford3dDataset_v1.2_Aligned_Version` now. If the original dataset is unzipped in a different location, then change the `DATA_PATH` variable in line 13 of `data_utils/collect_downsample.py`.
3. Run the dataset processing script. This script will collect each room from the original dataset, append label index, downsample it, format it as XYZRGBL and store it as `.npy` files. The processed `.npy` files will be stored inside `data/stanford_indoor3d_downsampled`.
```shell
cd data_utils
python collect_downsample.py 
```
This step need not be repeated again as long as the `.npy` are stored in the correct folder. Both the test and train scripts use these `.npy` files as input. Currently this script randomly samples 50% of the original pointcloud to downsample it. If you want to change it then change the variable `sampling_ratio` in line 95 of `data_utils/collect_downsample.py`.

## Train using S3DIS
Pre-trained models are available `log/sem_seg/pointnet2_sem_seg/`. There are three models trained on S3DIS. The model `pointnet2_sem_seg` was trained on all the 13 classes, `pointnet2_door_window` was trained on 4 classes i.e. 'door', 'window', 'wall' and other, and `pointnet2_best_door` was trained on 3 classes i.e. 'door', 'wall' and 'other'.

However, if model needs to be trained on S3DIS dataset then run the train script. Make sure you have followed the data preparation steps in the previous section. From root directory of PointNet2 i.e., `LiDAR_Semantic_Segmentation/PointNet2`, run the train script.

```shell
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --batch_size 32 --epoch 32
```
It is recommended to use a smaller batch size if training on laptop. If the train script fails with a "Killed" message then most likely it means it ran out of RAM memory, so reduce batch size in this case.


## Data preparation for LiDAR-Net dataset
1. Make sure you are in the root directory of PointNet2 i.e., `LiDAR_Semantic_Segmentation/PointNet2`. Create the following directory to store original dataset and processed dataset. 
```shell
mkdir data/
mkdir data/lidarnet_downsampled
```
2. Download the `Working Scenes` data from [here](http://lidar-net.njumeta.com/index.php/download/).
3. Unzip and store it inside `data/`. The directory structure should be `data/working` now. If the original dataset is unzipped in a different location, then change the `DATA_PATH` variable in line 10 of `data_utils/collect_lidarnet.py`.
3. Run the dataset processing script. This script will downsample and store the dataset as `.npy` in XYZRGBL format. The processed `.npy` files will be stored inside `data/lidarnet_downsampled`.
```shell
cd data_utils
python collect_lidarnet.py 
```
This script can take ~40 minutes to finish as the LiDAR-Net dataset is big. This step need not be repeated again as long as the `.npy` are stored in the correct folder. Both the test and train scripts use these `.npy` files as input. Currently this script randomly samples 10% of the original pointcloud to downsample it. If you want to change it then change the variable `sampling_ratio` in line 83 of `data_utils/collect_lidarnet.py`.

## Train using LiDAR-Net
Pre-trained model is available in `log/sem_seg/pointnet2_sem_seg_lidarnet/`. This model was trained on 3 classes i.e. 'door', 'wall' and 'other'. However, if model needs to be trained on LiDAR-Net dataset then run the train script. Make sure you have followed the data preparation steps in the previous section. From root directory of PointNet2 i.e., `LiDAR_Semantic_Segmentation/PointNet2`, run the train script.

```shell
python train_semseg_lidarnet.py --model pointnet2_sem_seg --batch_size 32 --epoch 32
```
It is recommended to use a smaller batch size if training on laptop. If the train script stops running with a "Killed" message then most likely it means it ran out of RAM memory, so reduce batch size in this case.

## Testing on custom data
Trained models can be tested on custom data without labels for a qualitative evaluation. The custom pointcloud data should have atleast XYZRGB features. XYZ needs to be in meters and RGB should be an integer from 0 to 255. The custom data needs to be pre-processed before running inference for which a pre-process script is provided. Some example point clouds are already provided in `test_data` for reference.

1. Save your custom data as a `.npy` file inside `test_data/`. If your data is in a different format, you will have to write a script to convert it into a `.npy` file with first 6 columns being XYZRGB.
2. Read the `pre_process.py` script to see how to load your custom data. Change the data path in the script and run it.
```shell
python pre_process.py
```
3. Trained models are available in `log/sem_seg'. Run the test script.
```shell
python test_semseg_custom.py --log_dir pointnet2_best_door --visual --batch_size 16
```
Results will be stored as a `.txt` file in `log/sem_seg/pointnet2_best_door/visual`. A 3D viewing tool like CloudCompare can be used to load and view this file. You can choose a different model from log/semseg with the --log_dir arguement. If the test scrip fails with a "Killed" message then reduce --batch_size.

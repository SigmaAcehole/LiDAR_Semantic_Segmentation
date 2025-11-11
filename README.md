# ROS2 Implementation
This README explains how to use ROS2 package made to locate doors and openings from Velodyne LiDAR point cloud. The implementation was made using computer vision algorithms. However, work done with the deep learning model i.e. PointNet++ is also provided. To understand how to train and test on it, refer `PointNet2/README.md`. 

![Visualization](Visualization.gif)  
*Demo showing real-time door and window detection from 3D LiDAR data*

## Tested Environment
1. OS: Ubuntu 22.04 LTS (Used WSL2 but dual-boot should work as well) 
2. Python 3.12.4
3. ROS2-humble
4. Open3D 0.18.0

## Setup the environment
Clone this repository.
```shell
git clone https://github.com/SigmaAcehole/LiDAR_Semantic_Segmentation.git
```
Make sure to have ROS2-humble installed. Instructions for installation can be found [here](https://control.ros.org/humble/doc/getting_started/getting_started.html). 

[Optional] To avoid dependency issues, its always a good idea to setup a virtual environment where the required python packages with the correct versions are uninstalled. This will prevent conflicts with other packages. Follow the following steps to setup the virtual environment using Miniconda.

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
4. Create a conda environment with Python 3.12.4
```shell
conda create -n ros2 python=3.12.4
```
5. Activate the environment.
```shell
conda activate ros2
``` 
6. Install Open3D.   
```shell 
conda install conda-forge::open3d
```

## How to run it
1. Install Open3D if you are not using virtual environment as mentioned above. Skip this step if you followed the steps for virtual environment.
```shell
pip install open3d
```
2. Build the package from the workspace.
```shell
cd ros2_ws
colcon build
```
3. Open another terminal at this directory and source the overlay.
```shell
source install/setup.bash
```
4. The ROS2 node can be run in two ways.    
To run it with real-time visualizations of the bounding boxes while publishing.
```shell
ros2 run lidar_seg cluster --ros-args -p visual:=1
```
To run it to just publish the bounding boxes but not visualize it.
```shell
ros2 run lidar_seg cluster
```
5. If the velodyne data is stored in a rosbag then play the rosbag.
```shell
ros2 bag play [bag_directory/bag_name.db3]
```
6. [OPTIONAL] The `cluster` node publishes the coordinates of 8 corner points of the bounding box as a PointCloud2 message. They are published in topic `/door` and `/opening` for detected doors and opening respectively. If no bounding box detected, a single point with coordinates [0,0,0] is published by default. Each bounding box is represented by an array of shape 8x3 so if two boxes are detected then the shape of the published point cloud will be 16x2. A node `test` is provided that subscribes to these bounding boxes and prints the number of doors and opening detected. This node can be used as a reference to use the results of `cluster` node in the future.     
On a new terminal, source and run the `test` node.
```shell
source install/setup.bash
ros2 run lidar_seg test
```

## Additional information and other useful node
A node `preproc` is provided which is not used in the traditional method. It subscribes to `\velodyne_points` publishes by Velodyne LiDAR or rosbag. It then does intensity based color transformation to add RGB details. It then uses a moving window ICP method to make the point cloud more dense. It publishes the point cloud of shape `n x 7` where `n` is the number of points and the 7 columns are RGB + XYZ + Intensity. This point cloud is published in topic `\xyz_rgb`. This can be useful when making inference on the point cloud with the deep learning model. 

1. To use, source a new terminal and run the node. 
```shell
source install/setup.bash
ros2 run lidar_seg preproc
```
2. Run the rosbag.
3. Results can be visualized using RViz. The frame id of the published point cloud is `rgb`. 

import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ubuntu/Internship/LiDAR_Semantic_Segmentation/ros2_ws/install/lidar_seg'

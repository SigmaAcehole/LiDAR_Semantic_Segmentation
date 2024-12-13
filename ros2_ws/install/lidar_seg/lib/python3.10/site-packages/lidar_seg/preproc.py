import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
from sensor_msgs_py import point_cloud2
import sys
from collections import namedtuple
import ctypes
import math
import struct
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import open3d as o3d
from collections import deque


class PreProc(Node):

    def __init__(self):
        super().__init__('preproc')

        # Subscriber
        self.subscription = self.create_subscription(
            PointCloud2,    # Msg type
            '/velodyne_points',                      # topic
            self.listener_callback,      # Function to call
            10                          # QoS
        )
        self.scan_num = 0
        self.subscription

        # Publisher
        self.publisher_ = self.create_publisher(PointCloud2, 'xyz_rgb', 10)
        timer_period = 1/20  # seconds
        self.timer = self.create_timer(timer_period, self.publisher_callback) 

        # Initializing parameters and objects for moving window point cloud registration
        self.is_first_scan = True 
        self.global_trans_done = False
        self.trans_init = np.ones((4,4))
        self.window = deque()   # Queue used to maintain the moving window
        self.window_size = 5    
        self.voxel_size = 0.05
        self.threshold_icp = 0.02
        self.pc = o3d.geometry.PointCloud()   # The final dense point cloud that will be published
        self.pc.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
        self.pc.colors = o3d.utility.Vector3dVector(np.zeros((1,3)))
        self.source = o3d.geometry.PointCloud() 
        self.target = o3d.geometry.PointCloud()

        self.intensity = np.zeros((1,1))


    # Callback for subscriber
    def listener_callback(self, msg):
        cloud = np.array(list(read_points(cloud= msg, field_names= ['x', 'y', 'z', 'intensity'])))    # Extract XYZ and intensity from incoming point cloud and store as numpy array

        points =  cloud[:,0:3]    # XYZ coordinates
        colors = np.zeros((cloud.shape[0], 3))      # RGB channels

        # Compute min and max intensity from cloud
        intensity_max = -9999
        intensity_min = 9999
        for i in range(cloud.shape[0]):
            intensity_max = max(intensity_max, cloud[i,3])
            intensity_min = min(intensity_min, cloud[i,3])
        intensity_max = min(9999, intensity_max)
        intensity_min = max(-9999, intensity_min)
        intensity_diff = intensity_max - intensity_min


        # Compute RGB colors based on intensity
        for j in range(cloud.shape[0]):
            intensity_norm = 1 - (cloud[j,3] - intensity_min)/intensity_diff     # Normalized intensity
            colors[j,0], colors[j,1], colors[j,2] = get_RGB(intensity_norm)      # Color mapping from normalized intensity to RGB

        # Initialize moving window for ICP registration of window size as 5 using first 5 scans received
        if(self.scan_num < self.window_size):
            if(self.is_first_scan):     # It is the first scan so no registration is done
                self.source.points = o3d.utility.Vector3dVector(points)
                self.window.append(len(points))
                self.intensity = cloud[:,3].reshape((len(points),1))  
                print(self.intensity.shape)
                self.pc.points = self.source.points
                self.pc.colors = o3d.utility.Vector3dVector(colors)
                self.is_first_scan = False
            else:       # Registration begins from second scan 
                self.target.points = o3d.utility.Vector3dVector(points)
                self.window.append(len(points))
                if(self.global_trans_done == False):
                    self.trans_init = fast_global_registration(self.source, self.target, self.voxel_size)
                    self.global_trans_done = False
                transformation_matrix = icp_transformation(self.source, self.target, self.trans_init, self.threshold_icp)
                self.pc.transform(transformation_matrix)
                self.pc.points.extend(self.target.points)
                self.pc.colors.extend(o3d.utility.Vector3dVector(colors))
                self.intensity = np.vstack((self.intensity, cloud[:,3].reshape((len(points),1))))
                self.source.points = self.target.points
        # Update the moving window
        else:
            self.target.points = o3d.utility.Vector3dVector(points)
            # Remove previous scan points and colors
            scan_remove = self.window.popleft()
            self.pc.points = o3d.utility.Vector3dVector(np.delete(np.asarray(self.pc.points),np.s_[0:scan_remove], axis=0))
            self.pc.colors = o3d.utility.Vector3dVector(np.delete(np.asarray(self.pc.colors),np.s_[0:scan_remove], axis=0))
            self.intensity = np.delete(self.intensity, np.s_[0:scan_remove], axis=0)
            self.window.append(len(points))
            if(self.global_trans_done == False):
                self.trans_init = fast_global_registration(self.source, self.target, self.voxel_size)
                self.global_trans_done = False
            transformation_matrix = icp_transformation(self.source, self.target, self.trans_init, self.threshold_icp)
            self.pc.transform(transformation_matrix)
            self.pc.points.extend(self.target.points)
            self.pc.colors.extend(o3d.utility.Vector3dVector(colors))
            self.intensity = np.vstack((self.intensity, cloud[:,3].reshape((len(points),1))))
            self.source.points = self.target.points
        
        self.scan_num += 1

    # Callback for publisher
    def publisher_callback(self):
        # Create PointCloud2 message with custom fields (X,Y,Z,R,G,B,Intensity)
        header = Header()
        fields =[PointField(name = 'x', offset = 0, datatype = 7, count = 1),
                PointField(name = 'y', offset = 4, datatype = 7, count = 1),
                PointField(name = 'z', offset = 8, datatype = 7, count = 1),
                PointField(name = 'r', offset = 12, datatype = 7, count = 1),
                PointField(name = 'g', offset = 16, datatype = 7, count = 1),
                PointField(name = 'b', offset = 20, datatype = 7, count = 1),
                PointField(name = 'intensity', offset = 24, datatype = 7, count = 1,)
                ]

        print("Scan no.: ", self.scan_num)
        print("Intensity shape: ", self.intensity.shape)
        # Publisher after moving window created
        if(self.scan_num >= self.window_size):       
            pointcloud_msg = point_cloud2.create_cloud(header, fields, np.concatenate((np.asarray(self.pc.points),np.asarray(self.pc.colors),self.intensity),axis=1))
            pointcloud_msg.header.frame_id = "rgb"
            self.publisher_.publish(pointcloud_msg)


def icp_transformation(source, target, trans_init, threshold):
    """
    Returns transformation matrix between two point clouds computed using ICP algorithm.
    @param source: The point cloud that needs to be transformed
    @type source: open3d.geometry.PointCloud()
    @param target: The point cloud with respect to which transformation is computed
    @type target: open3d.geometry.PointCloud()
    @param trans_init: A rough initial transformation matrix (can be computed using global alignment methods)
    @type trans_init: Numpy array of shape (4,4)
    @param threshold: Threshold for ICP in units of the XYZ coordinate system
    @type treshold: float 
    @return: The transformation matrix
    @rtype: Numpy array of shape (4,4)
    """
    result_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 50))
    
    # print(result_p2p)
    return result_p2p.transformation

def fast_global_registration(source, target, voxel_size):
    """
    Returns transformation matrix between two point clouds using fast global registration method. This is used as an initial approximate transformation by ICP.
    @param source: The point cloud that needs to be transformed
    @type source: open3d.geometry.PointCloud()
    @param target: The point cloud with respect to which transformation is computed
    @type target: open3d.geometry.PointCloud()
    @param voxel_size: Size of voxel used for downsampling, in same units as the XYZ coordinate system
    @type voxel_size: float
    @return: The transformation matrix
    @rtype: Numpy array of shape (4,4)
    """
    distance_threshold = voxel_size * 0.5
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size) 
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result.transformation

def preprocess_point_cloud(pcd, voxel_size):
    """
    Returns downsampled pointcloud and features.
    @param pcd: Input pointcloud
    @type pcd: open3d.geometry.PointCloud()
    @param voxel_size: Size of voxel used for downsampling, in same units as the XYZ coordinate system
    @type voxel_size: float
    @return: Downsampled pointcloud and extracted features
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

"""
The code for the function below is based on rviz's point cloud intensity based color transformer
Source (line 46): https://docs.ros.org/en/jade/api/rviz/html/c++/point__cloud__transformers_8cpp_source.html
"""
def get_RGB(val):
    """
    Returns RGB values from normalized intensity value.
    @param val: Normalized intensity value
    @type val: float
    @return: R, G and B values in range [0,1]
    @rtype: float  
    """
    # Make sure val is in range [0,1]
    val = min(val,1)
    val = max(val,0)
    # Use HSV color palette to get RGB
    h = val * 5 + 1
    i = math.floor(h)
    f = h - i
    if (i%2 == 0):
        f = 1 - f
    n = 1 - f
    if(i <= 1):
        r = n
        g = 0
        b = 1
    elif(i == 2):
        r = 0
        g = n 
        b = 1
    elif(i == 3):
        r = 0
        g = 1
        b = n
    elif(i == 4):
        r = n
        g = 1 
        b = 0
    elif(i >= 5):
        r = 1
        g = n 
        b = 0
    
    return r,g,b

"""
The code for function below to read and unpack PointCloud2 message is taken from:        
https://github.com/ros/common_msgs/blob/noetic-devel/sensor_msgs/src/sensor_msgs/point_cloud2.py
"""
_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

def read_points(cloud, field_names= [], skip_nans=True, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.

    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    assert isinstance(cloud, PointCloud2), 'cloud is not a sensor_msgs.msg.PointCloud2'
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step

def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt

def main(args=None):
    rclpy.init(args=args)

    subscriber = PreProc()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
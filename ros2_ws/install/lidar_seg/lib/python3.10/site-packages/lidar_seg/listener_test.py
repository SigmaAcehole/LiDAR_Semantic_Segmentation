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

height = 1
width = 6
xyzrgb_cloud = np.zeros((1,6))


class PCListener(Node):

    def __init__(self):
        super().__init__('listener_test')

        ##     Uncomment to initialize open3d visualization     ##
        ## Initialize open3d visualizer object and point cloud object
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window()
        # self.o3d_pcd = o3d.geometry.PointCloud()

        # Subscriber
        self.subscription = self.create_subscription(
            PointCloud2,    # Msg type
            '/velodyne_points',                      # topic
            self.listener_callback,      # Function to call
            10                          # QoS
        )
        self.subscription

        # Publisher
        self.publisher_ = self.create_publisher(PointCloud2, 'xyz_rgb', 10)
        timer_period = 1/20  # seconds
        self.timer = self.create_timer(timer_period, self.publisher_callback)
        
        # TODO: Initialize xyzrgb_cloud with correct shape  


    # Callback for subscriber
    def listener_callback(self, msg):
        self.get_logger().info('Received PointCloud2 message')
        cloud = np.array(list(read_points(msg)))
        height = msg.height
        width = msg.width

        points =  np.array(cloud[:,0:3])    # x, y, z coordinates
        colors = np.zeros((cloud.shape[0], 3))      # colors with r, g, b values

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
            intensity = 1 - (cloud[j,3] - intensity_min)/intensity_diff     # Normalized intensity
            colors[j,0], colors[j,1], colors[j,2] = get_RGB(intensity)

        # Create XYZRGB array
        global xyzrgb_cloud
        xyzrgb_cloud = np.hstack((points, colors))

        ##       Uncomment to visualize point cloud using open3d        ##
        # self.vis.remove_geometry(self.o3d_pcd)
        # self.o3d_pcd.points = o3d.utility.Vector3dVector(points)
        # self.o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
        # self.vis.add_geometry(self.o3d_pcd)
        # self.vis.poll_events()
        # self.vis.update_renderer()

    # Callback for publisher
    def publisher_callback(self):
        fields =[PointField(name = 'x', offset = 0, datatype = 7, count = 1),
                PointField(name = 'y', offset = 4, datatype = 7, count = 1),
                PointField(name = 'z', offset = 8, datatype = 7, count = 1),
                PointField(name = 'r', offset = 12, datatype = 7, count = 1),
                PointField(name = 'g', offset = 16, datatype = 7, count = 1),
                PointField(name = 'b', offset = 20, datatype = 7, count = 1),
                ]
        header = Header()
    
        global xyzrgb_cloud, height, width
        pointcloud_msg = point_cloud2.create_cloud(header, fields, xyzrgb_cloud)
        pointcloud_msg.header.frame_id = "rgb"
        self.publisher_.publish(pointcloud_msg)


# Get RGB values from normalized intensity value
# The code for the function is based on rviz's point cloud intensity based color transformer
# Source (line 46): https://docs.ros.org/en/jade/api/rviz/html/c++/point__cloud__transformers_8cpp_source.html
def get_RGB(val):
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

# The code for function below to read and unpack PointCloud2 message is taken from:        
# https://github.com/ros/common_msgs/blob/noetic-devel/sensor_msgs/src/sensor_msgs/point_cloud2.py
_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

def read_points(cloud, field_names= ['x', 'y', 'z', 'intensity'], skip_nans=True, uvs=[]):
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
    print(field_names)
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

    minimal_subscriber = PCListener()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
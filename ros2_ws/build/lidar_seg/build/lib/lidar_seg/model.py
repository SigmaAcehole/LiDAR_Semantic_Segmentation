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

class Model(Node):

    def __init__(self):
        super().__init__('model')

        # Subscriber
        self.subscription = self.create_subscription(
            PointCloud2,    # Msg type
            '/xyz_rgb',                      # topic
            self.listener_callback,      # Function to call
            10                          # QoS
        )
        self.scan_num = 0
        self.subscription

        # Publisher
        # self.publisher_ = self.create_publisher(PointCloud2, 'xyz_rgb', 10)
        # timer_period = 1/10  # seconds
        # self.timer = self.create_timer(timer_period, self.publisher_callback) 

        # Initializing parameters and objects for moving window point cloud registration
        

    # Callback for subscriber
    def listener_callback(self, msg):
        if(self.scan_num < 1):
            cloud = np.array(list(read_points(cloud= msg, field_names= ['x', 'y', 'z', 'r', 'g', 'b', 'intensity'])))    # Extract XYZ, RGB and intensity from incoming point cloud and store as numpy array
            np.save('lab_corridor_6.npy', cloud)
            fout = open("lab_corridor_6.txt", 'w')
            for i in range(cloud.shape[0]):
                    fout.write('%f %f %f %d %d %d %d\n' % (
                        cloud[i, 0], cloud[i, 1], cloud[i, 2], cloud[i, 3], cloud[i, 4],
                        cloud[i, 5], cloud[i, 6]))
            fout.close() 
            print(cloud.shape)
            self.scan_num +=1
        

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

    subscriber = Model()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
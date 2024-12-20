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
import copy
import matplotlib.pyplot as plt
from geometry_msgs.msg import PointStamped

class Cluster(Node):

    def __init__(self):
        super().__init__('cluster')
        self.declare_parameter('visual', 0) # setting to 1 will display Open3D visualization

        # Subscriber
        self.subscription = self.create_subscription(
            PointCloud2,    # Msg type
            '/velodyne_points',                      # topic
            self.listener_callback,      # Function to call
            10                          # QoS
        )
        self.scan_num = 0
        self.cloud = np.array([])
        self.subscription

        if(self.get_parameter('visual').value == 1):
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.o3d_pcd = o3d.geometry.PointCloud()

        # Publisher for door
        self.publisher_door_ = self.create_publisher(PointCloud2, '/door', 10)
        timer_period = 1/20 
        self.timer = self.create_timer(timer_period, self.publisher_callback_door) 

        # Publisher for opening
        self.publisher_opening_ = self.create_publisher(PointCloud2, '/opening', 10)
        timer_period = 1/20
        self.timer = self.create_timer(timer_period, self.publisher_callback_opening)

        self.bounding_box_door = []
        self.bounding_box_opening = []
        

    # Callback for subscriber
    def listener_callback(self, msg):
        self.cloud = np.array(list(read_points(cloud= msg, field_names= ['x', 'y', 'z', 'intensity'])))    # Extract XYZ, RGB and intensity from incoming point cloud and store as numpy array 
        door_threshold = 40
        self.bounding_box_door, self.bounding_box_opening = segment_doors_opening(self.cloud, door_threshold)

        if(self.get_parameter('visual').value == 1):
            self.vis.clear_geometries()
            self.o3d_pcd.points = o3d.utility.Vector3dVector(self.cloud[:,:3])
            self.o3d_pcd.paint_uniform_color([0,0,0])
            if(self.scan_num == 0):
                self.vis.add_geometry(self.o3d_pcd)
            else:
                self.vis.update_geometry(self.o3d_pcd)
            for i in range(len(self.bounding_box_door)):
                self.vis.add_geometry(self.bounding_box_door[i])
            for j in range(len(self.bounding_box_opening)):
                self.vis.add_geometry(self.bounding_box_opening[j])   
            self.vis.get_render_option().point_size = 2
            self.vis.get_render_option().light_on = False
            self.vis.get_view_control().set_zoom(0.2)
            self.vis.get_view_control().change_field_of_view(100)
            self.vis.poll_events()
            self.vis.update_renderer()

    
    def publisher_callback_door(self):
        header = Header()
        fields =[PointField(name = 'x', offset = 0, datatype = 7, count = 1),
            PointField(name = 'y', offset = 4, datatype = 7, count = 1),
            PointField(name = 'z', offset = 8, datatype = 7, count = 1)
            ]

        bb_points_all = np.zeros((1,3))
        if self.bounding_box_door:
            bb_points_all = np.asarray(self.bounding_box_door[0].get_box_points())
            for i in range(len(self.bounding_box_door)-1):
                bb_points_all = np.append(bb_points_all, np.asarray(self.bounding_box_door[i+1].get_box_points()), axis=0)
            
        pointcloud_msg = point_cloud2.create_cloud(header, fields, bb_points_all)
        pointcloud_msg.header.frame_id = "door"
        self.publisher_door_.publish(pointcloud_msg)

    def publisher_callback_opening(self):
        header = Header()
        fields =[PointField(name = 'x', offset = 0, datatype = 7, count = 1),
            PointField(name = 'y', offset = 4, datatype = 7, count = 1),
            PointField(name = 'z', offset = 8, datatype = 7, count = 1)
            ]

        bb_points_all = np.zeros((1,3))
        if self.bounding_box_opening:
            bb_points_all = np.asarray(self.bounding_box_opening[0].get_box_points())
            for i in range(len(self.bounding_box_opening)-1):
                bb_points_all = np.append(bb_points_all, np.asarray(self.bounding_box_opening[i+1].get_box_points()), axis=0)
            
        pointcloud_msg = point_cloud2.create_cloud(header, fields, bb_points_all)
        pointcloud_msg.header.frame_id = "opening"
        self.publisher_opening_.publish(pointcloud_msg)

        
def segment_doors_opening(data, door_threshold):
    # Convert numpy array to Open3d point cloud
    intensity = np.array([data[:,3], np.zeros(data.shape[0]), np.zeros(data.shape[0])]).T
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data[:,:3])
    pc.normals = o3d.utility.Vector3dVector(intensity)  # intensity is stored in first column of normals, rest two columns are dummy
    
    # Segment wall plane using RANSAC
    pc_ransac, plane = get_plane(pc)

    # Cluster using DBSCAN
    pc_cluster, labels = compute_cluster(pc_ransac)

    # Initialize variables to locate doors and opening
    num_clusters =  max(labels) + 1     # Number of clusters found   
    bounding_box_all = []               # Bounding box for each cluster, used to locate openings 
    bounding_box_door = []              # Bounding box on located door
    bounding_box_opening = []           # Bounding box on located opening
    door_num_points = 500               # Minimum number of points to be considered a door (to reduce false cases)
    plane_threshold = 0.1               # Maximum distance between a point and a plane to be considered within the plane   

    plane1_dict, plane2_dict = {}, {}   # Empty dictionary, Key : Value => bounding box : x_center/y_center               

    # Check whether the planes align closely with XZ or XY plane
    is_xz_plane1 = True if (plane[1] > plane[0]) else False
    is_xz_plane2 = True if (plane[5] > plane[4]) else False

    # Loop through each cluster
    for i in range(num_clusters):
        ind = np.where(labels == i)[0]
        cluster = pc_cluster.select_by_index(ind)
        x,y,z = cluster.get_center()    # Center of cluster
        bounding_box = rotate_bb_using_eigen(cluster)
        bounding_box.color = [0,1,0]
        # Check whether cluster is in plane 1 or plane 2
        plane1_dist = (np.abs(np.dot(plane[:3],np.array([x,y,z])) + plane[3]))/(np.linalg.norm(plane[:3]))
        if(plane1_dist <= plane_threshold): 
            plane1_dict[bounding_box] =  x if is_xz_plane1 else y
        else:
            plane2_dict[bounding_box] =  x if is_xz_plane2 else y
        bounding_box_all.append(bounding_box)
        # Check and locate door
        door_cluster = locate_door(cluster,door_threshold)
        if(len(door_cluster.points) > door_num_points):
            door_bound = rotate_bb_using_eigen(door_cluster)
            door_bound.color = [1.,0.,0.]
            bounding_box_door.append(door_bound)
      
    # Sort arrangement of clusters inside plane in ascending order based on x/y center value
    plane1_sorted = [key for key, val in sorted(plane1_dict.items(), key = lambda ele: ele[1])]
    plane2_sorted = [key for key, val in sorted(plane2_dict.items(), key = lambda ele: ele[1])]

    # Locate openings in each plane
    plane1_len = len(plane1_sorted)
    plane2_len = len(plane2_sorted)
    if(plane1_len > 1):
        for m in range(len(plane1_sorted)-1):
            opening_box = locate_opening(plane1_sorted[m], plane1_sorted[m+1], is_xz_plane1)
            if(opening_box is not False):
                opening_box.color = [0., 0., 1.]
                bounding_box_opening.append(opening_box)
    if(plane2_len > 1):
        for m in range(len(plane2_sorted)-1):
            opening_box = locate_opening(plane2_sorted[m], plane2_sorted[m+1], is_xz_plane2)
            if(opening_box is not False):
                opening_box.color = [0., 0., 1.]
                bounding_box_opening.append(opening_box)

    return bounding_box_door, bounding_box_opening

def rotate_bb_using_eigen(pc):
    obox = pc.get_minimal_oriented_bounding_box()
    obox.color = [1,0,0]
    return obox

def locate_door(pc, door_threshold):
    cloud = np.hstack([pc.points, pc.normals])[:,:4]
    avg_intensity = np.average(cloud[:,3])
    if(avg_intensity <= door_threshold):
        return pc
    else:
        mask = np.ones(cloud.shape[0], dtype=bool)
        for i in range(cloud.shape[0]):
            if(cloud[i,3] > door_threshold/4):
                mask[i] = False
        # Extract points from cluster which have intensity lower than threshold
        cloud =  cloud[mask,:]
        door_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud[:,:3]))
        door_cloud.paint_uniform_color([1.,0.,0.])
        # Remove outliers
        _, ind = door_cloud.remove_radius_outlier(nb_points=16, radius=0.1)
        inlier_cloud = door_cloud.select_by_index(ind)
        inlier_cloud.paint_uniform_color([0.,0.,1.])
        return inlier_cloud

def locate_opening(box1, box2, is_xz_plane):
    min_bound_1 = box1.get_min_bound()
    max_bound_1 = box1.get_max_bound()
    min_bound_2 = box2.get_min_bound()
    max_bound_2 = box2.get_max_bound()
    if(is_xz_plane):
        # Check if adjacent bounding boxes are overlapping
        if(min_bound_2[0] < max_bound_1[0]):
            return False
        center = [(max_bound_1[0] + min_bound_2[0])/2, (max_bound_1[1] + min_bound_1[1])/2, (max_bound_1[2] + min_bound_1[2])/2]
        width = min_bound_2[0] - max_bound_1[0]
        height = max_bound_1[2] - min_bound_1[2]
        thick = max_bound_1[1] - min_bound_1[1]
    else:
        # Check if adjacent bounding boxes are overlapping
        if(min_bound_2[1] < max_bound_1[1]):
            return False
        center = [(max_bound_1[0] + min_bound_1[0])/2, (max_bound_1[1] + min_bound_2[1])/2, (max_bound_1[2] + min_bound_1[2])/2]
        width = max_bound_1[0] - min_bound_1[0]
        height = max_bound_1[2] - min_bound_1[2]
        thick = min_bound_2[1] - max_bound_1[1]
    box = o3d.geometry.OrientedBoundingBox()
    box.center = center
    box.extent = [width, thick, height]
    box.color = [0,0,1]
    return box

def compute_cluster(pc):
    # Downsample
    pc = pc.uniform_down_sample(2)
    # Cluster using DBSCAN
    labels = pc.cluster_dbscan(eps=0.35, min_points=100, print_progress=False)
    labels = np.asarray(labels)
    max_label = max(labels)
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors = np.asarray(colors[:,:3])   # RGB range 0 - 255
    colors[labels < 0] = 0  # Points that couldn't be clustered have label value -1 so make their color black
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc, labels
    
def get_plane(pc):
    # Plane segmentation using RANSAC
    distance_threshold = 0.08
    [a1,b1,c1,d1], inliers1 = pc.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=5000, probability=0.9999)
    inlier_cloud1 = pc.select_by_index(inliers1)
    outlier_cloud = pc.select_by_index(inliers1, invert=True)
    # Extract second plane using RANSAC
    [a2,b2,c2,d2], inliers2 = outlier_cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=5000, probability=0.9999)
    inlier_cloud2 = outlier_cloud.select_by_index(inliers2)
    if(c1 < 0.5 and c2 < 0.5):
        pc_ransac = copy.deepcopy(inlier_cloud1)
        pc_ransac.points.extend(inlier_cloud2.points)
        pc_ransac.normals.extend(inlier_cloud2.normals)
    elif(c1 < 0.5 and c2 >= 0.5):
        pc_ransac = copy.deepcopy(inlier_cloud1)
        a2,b2,c2,d2 = 0,0,1,9999
    else:
        pc_ransac = copy.deepcopy(inlier_cloud2)
        a1,b1,c1,d1 = 0,0,1,9999 
    return pc_ransac, np.array([a1,b1,c1,d1,a2,b2,c2,d2])

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

    subscriber = Cluster()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
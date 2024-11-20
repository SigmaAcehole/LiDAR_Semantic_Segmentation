import os
import sys
import glob
import open3d as o3d
import numpy as np
from plyfile import PlyData


def main():
    DATA_PATH = '../../../PointNet2/lidarnet/working/'

    # Store all file names in a list
    ply_files = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".ply"):
            ply_files.append(file[:-4])  # Remove the .ply extension
    
    output_folder = '../data/lidarnet_downsampled'

    # Check if output folder was created
    if not os.path.exists(output_folder):
        error_string = "Directory " + str(output_folder) + " does not exist, please create it!"
        raise ValueError(error_string)

    for file_name in ply_files:
        print(file_name)
        try:
            file_path = DATA_PATH + file_name + '.ply'
            out_filename = file_name + '.npy'   # Ex: baiyin_room07_part_01.npy
            collect_point_label(file_path, os.path.join(output_folder, out_filename), 'numpy')
        except:
            print(file_path, 'ERROR!!')

    print("Finished preparing dataset!")


def collect_point_label(file_path, out_filename, file_format='txt'):
    """ Convert original dataset files to data_label file (each line is XYZRGBL).
    Args:
        file_path: path to .ply file
        out_filename: path to save collected points and labels (each line is XYZRGBL)
        file_format: txt or numpy, determines what file format to save.
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """
    data_label = read_ply_as_numpy(file_path)
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min   # Shift most negative point to origin

    # Downsample the pointcloud
    data_label = down_sample(data_label)

    if file_format=='txt':
        fout = open(out_filename, 'w')
        for i in range(data_label.shape[0]):
            fout.write('%f %f %f %d %d %d %d\n' % \
                          (data_label[i,0], data_label[i,1], data_label[i,2],
                           data_label[i,3], data_label[i,4], data_label[i,5],
                           data_label[i,6]))
        fout.close()
    elif file_format=='numpy':
        np.save(out_filename, data_label)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
            (file_format))
        exit()

def down_sample(data):
    """Reads a pointcloud and downsamples it.
    Args:
        data: Pointcloud as a numpy array.
    Returns:
        Downsampled pointcloud as numpy array.
    """
    labels = np.array([data[:,6], np.zeros(data.shape[0]), np.zeros(data.shape[0])]).T    
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data[:,:3])
    pc.colors = o3d.utility.Vector3dVector(data[:,3:6])
    pc.normals = o3d.utility.Vector3dVector(labels)   # Storing labels as normals as open3d PointCloud() has no labels
    sampling_ratio = 0.1    # no. of sampled points/total no. of points. To sample all points keep it as 1.
    pc_down = pc.random_down_sample(sampling_ratio)
    points = np.asarray(pc_down.points)
    colors = np.asarray(pc_down.colors)
    labels = np.asarray(pc_down.normals)
    data = np.hstack((points,colors,labels))[:,:7]      # Back to XYZRGBL format
    return data

def read_ply_as_numpy(ply_file_path):
  """Reads a .ply file and returns its data as a NumPy array.
  Args:
    ply_file_path: Path to the .ply file.
  Returns:
    A NumPy array containing the data from the .ply file.
  """
  plydata = PlyData.read(ply_file_path)
  data = plydata['vertex'].data
  # Extract relevant data (X Y Z R G B L)
  x, y, z, r, g, b, l = data['x'], data['y'], data['z'], data['red'], data['green'], data['blue'], data['sem']
  # Create a NumPy array
  point_cloud = np.array([x, y, z, r, g, b, l]).transpose()

  return point_cloud

if __name__ == '__main__':
    main()

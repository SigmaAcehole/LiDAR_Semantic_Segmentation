import os
import sys
import glob
import open3d as o3d
import numpy as np

def main():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)

    # Manually set the path to original dataset
    # DATA_PATH = '../../../PointNet2/s3dis/Stanford3dDataset_v1.2_Aligned_Version/'
    DATA_PATH = '/mnt/datasets/Eshan/Stanford3dDataset_v1.2_Aligned_Version'

    anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
    anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

    output_folder = '../data/stanford_indoor3d_downsampled'

    # Check if output folder was created
    if not os.path.exists(output_folder):
        error_string = "Directory " + str(output_folder) + " does not exist, please create it!"
        raise ValueError(error_string)
    

    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    for anno_path in anno_paths:
        print(anno_path)
        try:
            elements = anno_path.split('/')
            out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Ex: Area_1_hallway_1.npy
            collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
        except:
            print(anno_path, 'ERROR!!')

    print("Finished preparing dataset!")
    

def collect_point_label(anno_path, out_filename, file_format='txt'):
    """ Convert original dataset files to data_label file (each line is XYZRGBL).
        We aggregated all the points from each instance in the room.

    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBL)
        file_format: txt or numpy, determines what file format to save.
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    g_classes = [x.rstrip() for x in open(os.path.join(BASE_DIR, 'meta/class_names.txt'))]
    g_class2label = {cls: i for i,cls in enumerate(g_classes)}

    points_list = []
    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        print(f)
        if cls not in g_classes: # note: in some room there is 'stairs' class..
            cls = 'clutter'

        points = np.loadtxt(f)
        labels = np.ones((points.shape[0],1)) * g_class2label[cls]
        points_list.append(np.concatenate([points, labels], 1)) # Nx7
    
    data_label = np.concatenate(points_list, 0)
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min

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
    labels = np.array([data[:,6], np.zeros(data.shape[0]), np.zeros(data.shape[0])]).T    
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data[:,:3])
    pc.colors = o3d.utility.Vector3dVector(data[:,3:6])
    pc.normals = o3d.utility.Vector3dVector(labels)   # Storing labels as normals as open3d PointCloud() has no labels
    sampling_ratio = 0.3    # no. of sampled points/total no. of points. To sample all points keep it as 1.
    pc_down = pc.random_down_sample(sampling_ratio)
    points = np.asarray(pc_down.points)
    colors = np.asarray(pc_down.colors)
    labels = np.asarray(pc_down.normals)
    data = np.hstack((points,colors,labels))[:,:7]      # Back to XYZRGBL format
    return data

if __name__ == '__main__':
    main()

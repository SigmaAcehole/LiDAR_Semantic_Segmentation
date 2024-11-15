import numpy as np
import open3d as o3d 
from data_utils.indoor3d_util import g_label2color
import matplotlib.pyplot as plt

def main():
    data = np.load("test_data/segmented_scene.npy")
    print(data.shape)
    # Remove clutter
    indices_remove = []
    for i in range(data.shape[0]):
        if (data[i,-1] == 2 or data[i,-1] == 5 or data[i,-1] == 6):
            continue
        indices_remove.append(i)    
    
    print(len(indices_remove))
    data = np.delete(data, indices_remove, 0)
    print(data.shape)


    # Cluster
    data_cluster, labels = cluster(data)

    fout = open("test_data/segmented_scene_clustered.txt", 'w')
    for i in range(data_cluster.shape[0]):
            fout.write('%f %f %f %d %d %d\n' % (
                data_cluster[i, 0], data_cluster[i, 1], data_cluster[i, 2], data_cluster[i, 3], data_cluster[i, 4],
                data_cluster[i, 5]))
    fout.close() 

def cluster(data):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data[:,:3])
    pc.colors = o3d.utility.Vector3dVector(data[:,3:6])
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = pc.cluster_dbscan(eps=0.4, min_points=30, print_progress=True)
    labels = np.asarray(labels)
    max_label = max(labels)
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors = np.asarray(255*colors[:,:3])   # RGB range 0 - 255
    colors[labels < 0] = 0  # Points that couldn't be clustered have label value -1 so make their color black
    pc.colors = o3d.utility.Vector3dVector(colors)

    return np.hstack([np.asarray(pc.points), np.asarray(pc.colors)]), labels

if __name__ == '__main__':
    main()


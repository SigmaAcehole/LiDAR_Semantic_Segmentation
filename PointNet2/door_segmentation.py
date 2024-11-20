import numpy as np
import open3d as o3d 
import copy
import matplotlib.pyplot as plt
import time

def main():
    # Load npy file and extract XYZRGBI
    print("Loading data...")
    data = np.load("test_data/lab_corridor_2.npy")
    print("Data = ", data.shape)
    
    start = time.time()
    data_cluster, door_center = segment_doors(data)
    print("Time(s) = %.3f" %(time.time() - start))

    print("Number of doors: ", len(door_center)-1)
    print(door_center)

    # fout = open("test_data/segmented_door_clustered.txt", 'w')
    # for i in range(data_cluster.shape[0]):
    #         fout.write('%f %f %f %d %d %d\n' % (
    #             data_cluster[i, 0], data_cluster[i, 1], data_cluster[i, 2], data_cluster[i, 3], data_cluster[i, 4],
    #             data_cluster[i, 5]))
    # fout.close() 

def compute_cluster(pc):
    # Downsample
    pc = pc.uniform_down_sample(3)
    # Cluster using DBSCAN
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = pc.cluster_dbscan(eps=0.4, min_points=250, print_progress=False)
    labels = np.asarray(labels)
    max_label = max(labels)
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors = np.asarray(255*colors[:,:3])   # RGB range 0 - 255
    colors[labels < 0] = 0  # Points that couldn't be clustered have label value -1 so make their color black
    pc.colors = o3d.utility.Vector3dVector(colors)

    return pc, labels
    

def get_plane(pc):
    # Plane segmentation using RANSAC
    _, inliers = pc.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=5000, probability=0.9999)
    inlier_cloud = pc.select_by_index(inliers)
    outlier_cloud = pc.select_by_index(inliers, invert=True)
    _, inliers2 = outlier_cloud.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=5000, probability=0.9999)
    inlier_cloud2 = outlier_cloud.select_by_index(inliers2)
    pc_ransac = copy.deepcopy(inlier_cloud)
    pc_ransac.point.positions.extend(inlier_cloud2.point.positions)
    pc_ransac.point.colors.extend(inlier_cloud2.point.colors)
    pc_ransac.point.normals.extend(inlier_cloud2.point.normals)

    return pc_ransac

def segment_doors(data):
    dtype = o3d.core.float32
    pc = o3d.t.geometry.PointCloud()
    intensity = np.array([data[:,6], np.zeros(data.shape[0]), np.zeros(data.shape[0])]).T
    # pc.point.positions = o3d.utility.Vector3dVector(data[:,:3])
    # pc.colors = o3d.utility.Vector3dVector(data[:,3:6])
    # pc.normals = o3d.utility.Vector3dVector(intensity)
    pc.point.positions = o3d.core.Tensor(data[:,:3], dtype)
    pc.point.colors = o3d.core.Tensor(data[:,3:6], dtype)
    pc.point.normals = o3d.core.Tensor(intensity, dtype)

    # Segment wall plane using RANSAC
    pc_ransac = get_plane(pc)

    # Cluster using DBSCAN
    pc_cluster, labels = compute_cluster(pc_ransac)

    # Calculate average intensity of each cluster
    avg_intensity_list = []
    door_center = [[]]
    num_clusters = max(labels) + 1
    new_colors = np.zeros((len(pc_cluster.points),3))

    for i in range(num_clusters):
        ind = np.where(labels == i)[0]
        cluster = pc_cluster.select_by_index(ind)
        avg_intensity = np.average(np.asarray(cluster.normals)[:,0])
        avg_intensity_list.append(avg_intensity)

        print("Cluster ", i)
        print(" Points = %d" % (len(ind)))
        print(" Intensity= %f" % (avg_intensity))

        # Get center of door pointcloud and highlight doors with red
        if(avg_intensity < 40):
            (x,y,z) = cluster.get_center()
            door_center.append([x,y,z])
            new_colors[ind,:] = (255,0,0)

        # Detect boundary of cluster
        boundarys, mask = cluster.compute_boundary_points(0.02, 30)
        boundarys = boundarys.paint_uniform_color([1.0, 0.0, 0.0])
        print("Boundary = ", len(boundarys.points))
        print("Mask = ", len(mask))
            
    pc_cluster.colors = o3d.utility.Vector3dVector(new_colors)
    data_segmented = np.hstack([np.asarray(pc_cluster.points), np.asarray(pc_cluster.colors)])

    return data_segmented, door_center
    

if __name__ == '__main__':
    main()

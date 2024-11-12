import argparse
import os
# from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

import open3d as o3d 
import time
import copy

def main():
    # Load npy file and extract XYZRGB
    data = np.load("test_data/lab_corridor.npy")

    # # Save as .txt
    # data[:,3:6] = data[:,3:6] * 255
    # fout = open('test_data/lab_corridor_4.txt', 'w')
    # for i in range(data.shape[0]):
    #     fout.write('%f %f %f %f %f %f\n' % \
    #                 (data[i,0], data[i,1], data[i,2],
    #                 data[i,3], data[i,4], data[i,5]))
    # fout.close()


    # Pre-process it using  __getitem__(self, index) of S3DISDataLoader.py
        # Input is X+Y+Z+R+G+B
        # Output is X+Y+Z+R+G+B+nX+nY+nZ

    print(data.shape)
    
    # data = get_plane(data)
    # print(data.shape)
    np.save('lab_corridor_processed.npy', data)

    data_room, index_room = prepare_data(data, block_points=4096)
    np.save("scene_data.npy", data_room)
    np.save("scene_point_index.npy", index_room)
    print(data_room.shape)
    print(index_room.shape)

def prepare_data(points, block_points=4096):
    stride = 1
    block_size=1.0
    padding=0.001

    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - block_size) / stride) + 1)
    grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - block_size) / stride) + 1)

    data_room, index_room = np.array([]), np.array([])

    for index_y in range(0, grid_y):
        for index_x in range(0, grid_x):
            s_x = coord_min[0] + index_x * stride
            e_x = min(s_x + block_size, coord_max[0])
            s_x = e_x - block_size
            s_y = coord_min[1] + index_y * stride
            e_y = min(s_y + block_size, coord_max[1])
            s_y = e_y - block_size
            point_idxs = np.where(
                (points[:, 0] >= s_x - padding) & (points[:, 0] <= e_x + padding) & (points[:, 1] >= s_y - padding) & (
                            points[:, 1] <= e_y + padding))[0]
            if point_idxs.size == 0:
                continue
            num_batch = int(np.ceil(point_idxs.size / block_points))
            point_size = int(num_batch * block_points)
            replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
            point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
            point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
            np.random.shuffle(point_idxs)
            data_batch = points[point_idxs, :]
            normlized_xyz = np.zeros((point_size, 3))
            normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
            normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
            normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
            data_batch[:, 0] = data_batch[:, 0] - (s_x + block_size / 2.0)
            data_batch[:, 1] = data_batch[:, 1] - (s_y + block_size / 2.0)
            # data_batch[:, 3:6] /= 255.0
            data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)

            data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
            index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
     
    data_room = data_room.reshape((-1, block_points, data_room.shape[1]))
    index_room = index_room.reshape((-1, block_points))
    return data_room, index_room

def add_vote(vote_label_pool, point_idx, pred_label):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def get_plane(data):

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data[:,:3])
    pc.colors = o3d.utility.Vector3dVector(data[:,3:6])
    # Plane segmentation using RANSAC
    plane_model, inliers = pc.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=5000, probability=0.9999)
    [a,b,c,d] = plane_model
    # print(f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    inlier_cloud = pc.select_by_index(inliers)
    outlier_cloud = pc.select_by_index(inliers, invert=True)

    plane_model2, inliers2 = outlier_cloud.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=5000, probability=0.9999)
    [a2,b2,c2,d2] = plane_model
    # print(f"Plane equation: {a2:.3f}x + {b2:.3f}y + {c2:.3f}z + {d2:.3f} = 0")
    inlier_cloud2 = outlier_cloud.select_by_index(inliers2)
    # outlier_cloud2 = outlier_cloud.select_by_index(inliers2, invert=True)

    pc_ransac = copy.deepcopy(inlier_cloud)
    pc_ransac.points.extend(inlier_cloud2.points)
    pc_ransac.colors.extend(inlier_cloud2.colors)

    # voxel_down = pc_ransac.voxel_down_sample(voxel_size=0.04)
    # l, ind = voxel_down.remove_radius_outlier(nb_points=20, radius=0.165)
    # inlier_cloud3 = voxel_down.select_by_index(ind)


    return np.hstack([np.asarray(pc_ransac.points), np.asarray(pc_ransac.colors)])

if __name__ == '__main__':
    main()

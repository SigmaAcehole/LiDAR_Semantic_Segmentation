"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in testing [default: 4]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 3]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 4
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    print("model name = ", model_name)
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    #classifier = MODEL.get_model(NUM_CLASSES)
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()


    with torch.no_grad():
        log_string('---- EVALUATION WHOLE SCENE----')

        scene_data = np.load('scene_data.npy')
        scene_point_index = np.load('scene_point_index.npy')
        whole_scene_data = np.load('lab_corridor_processed.npy')

        print("scene_data = ", scene_data.shape)
        print("scene_point_index = ", scene_point_index.shape)
        print("whole_scene_data = ", whole_scene_data.shape)

        vote_label_pool = np.zeros((whole_scene_data.shape[0], NUM_CLASSES))
        for _ in tqdm(range(args.num_votes), total=args.num_votes):
            start = time.time()
            # scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
            # print("scene_data = ", scene_data.shape)
            # print("scene_label = ", scene_label.shape)
            # print("scene_smpw = ", scene_smpw.shape)
            # print("scene_point_index = ", scene_point_index.shape)
            num_blocks = scene_data.shape[0]
            print("num_blocks = ", num_blocks)
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            print("s_batch_num = ", s_batch_num)
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

            # batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
            # batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

            for sbatch in range(s_batch_num):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                # batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                # batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                batch_data[:, :, 3:6] /= 1.0

                torch_data = torch.Tensor(batch_data)
                torch_data = torch_data.float().cuda()
                #torch_data = torch_data.float()
                torch_data = torch_data.transpose(2, 1)
                print("torch_data = ", torch_data.shape) # torch.Size([4, 9, 4096])
                seg_pred, _ = classifier(torch_data)
                batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                            batch_pred_label[0:real_batch_size, ...])
                
                # print("batch_pred_label: ", batch_pred_label[:,:10])
                # print("batch_pred_label = ", batch_pred_label.shape) # (4, 4096)
                # print("real_batch_pred_label = ", batch_pred_label[0:real_batch_size, ...].shape)
                # print("vote_label_pool = ", vote_label_pool.shape)
                # print("real_batch_size = ", real_batch_size)
                # print("batch_pred_idx = ", batch_point_index[0:real_batch_size, ...])

                # pred_label[start_idx:end_idx]

            pred_label = np.argmax(vote_label_pool, 1)
            print("pred_label = ", pred_label.shape)
            print("Time %.3f sec.\n" % (time.time() - start))

        name = 'lab_1_ransac'

        if args.visual:
            fout = open(os.path.join(visual_dir, name + '_pred.txt'), 'w')

        filename = os.path.join(visual_dir, name + '.txt')
        with open(filename, 'w') as pl_save:
            for i in pred_label:
                pl_save.write(str(int(i)) + '\n')
            pl_save.close()
        for i in range(whole_scene_data.shape[0]):
            color = g_label2color[pred_label[i]]
            if args.visual:
                fout.write('%f %f %f %d %d %d\n' % (
                    whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                    color[2]))

        if args.visual:
            fout.close()

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)

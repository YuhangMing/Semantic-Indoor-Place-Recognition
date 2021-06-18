#
#
#      0=================================0
#      |       Generate Triple and       |
#      |   Quadruplet for VLAD layers    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Generate Triplets and Quadruplets for VLAD layers
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Yuhang Ming
#

import os
import pickle
import time
import numpy as np
from numpy.core.numeric import ones
import open3d as o3d

from sklearn.neighbors import KDTree
from utils.ply import read_ply, write_ply
from datasets.common import grid_subsampling

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # # Print New Line on Complete
    # if iteration == total: 
    #     print()


if __name__ == '__main__':
    scannet_path = '/media/yohann/Datasets/datasets/ScanNet'
    data_split_path = os.path.join(scannet_path, 'tools/Tasks/Benchmark')
    scannet_data_path = '/media/yohann/Datasets/datasets/ScanNet/scans'
    tasks = ['training', 'validation', 'test']
    # for scene in os.listdir(scannet_data_path):
    #     if 'scene' in scene:

    if not os.path.exists(os.path.join(scannet_data_path, 'input_pcd')):
        os.makedirs(os.path.join(scannet_data_path, 'input_pcd'))

    for task in tasks:
        print()
        if task == 'training':
            scene_file_name = os.path.join(data_split_path, 'scannetv2_train.txt')
            scenes = np.sort(np.loadtxt(scene_file_name, dtype=str))
        elif task == 'validation':
            scene_file_name = os.path.join(data_split_path, 'scannetv2_val.txt')
            scenes = np.sort(np.loadtxt(scene_file_name, dtype=str))
        else:
            scene_file_name = os.path.join(data_split_path, 'scannetv2_test.txt')
            scenes = np.loadtxt(scene_file_name, dtype=str)
        
        
        # test room numbers
        rooms = []
        for scene in scenes:
            room = scene[:9]
            if not room in rooms:
                rooms.append(room)
        print(task, 'has', len(rooms), 'scenes')

        # # start new file
        # vlad_file = open(os.path.join(scannet_path, 'vlad_'+task+'.txt'), 'w')
        # vlad_file.close()

        # data holders
        # # key = scene string, val = ids of all used frames (at step=15)
        # all_scene_frame_ids = {}
        # key = scene string, val = list of pairs of (list of pos, list of neg)
        all_scene_pos_neg = {}
        thre_dist = 2
        thre_ovlp = 0.2

        t = time.time()
        # loop through the scenes
        for scene in scenes:
            scannet_scene_path = os.path.join(scannet_data_path, scene)
            print(scannet_scene_path)
            
            scannet_pcd_path = os.path.join(scannet_data_path, 'input_pcd', scene)
            if not os.path.exists(scannet_pcd_path):
                os.makedirs(scannet_pcd_path)

            # get num of frames and the intrinsic matrix
            K = []
            nFrames = 0
            info_file = open(os.path.join(scannet_scene_path, scene+'.txt'))
            for line in info_file:
                vals = line.strip().split(' ')

                if 'depth' in vals[0]:
                    K.append(float(vals[2]))
                if vals[0] == 'numDepthFrames':
                    nFrames = int(vals[2])

            # load the high definition point cloud
            hd_scene_pcd = o3d.io.read_point_cloud(os.path.join(scannet_scene_path, scene+'_vh_clean.ply'))
            # get intrinsics into matrix
            intrinsics = o3d.camera.PinholeCameraIntrinsic(int(K[1]), int(K[0]), K[2], K[3], K[4], K[5])

            # loop through the frames to get pcd and cntr
            stepFrm = 15
            scene_ctrs = []
            scene_pcds = []
            scene_fids = []
            for i in range(0, nFrames, stepFrm):
                printProgressBar(i+1, nFrames, prefix=task+' - '+scene, suffix='Complete', length=50)
                
                # check if pose is lost
                pose = np.loadtxt(os.path.join(scannet_scene_path, 'pose', str(i)+'.txt'))
                chk_val = np.sum(pose)
                if np.isinf(chk_val) or np.isnan(chk_val):
                    continue

                # back project the raw depth 
                depth = o3d.io.read_image(os.path.join(scannet_scene_path, 'depth', str(i)+'.png'))
                depth_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics)

                # get aligned bounding box
                bbox_aligned = depth_pcd.get_axis_aligned_bounding_box()
                # crop pcd of interest
                cam_pcd = hd_scene_pcd.transform(np.linalg.inv(pose)).crop(bbox_aligned)

                # voxel grid downsample the point cloud
                cam_pts = np.asarray(cam_pcd.points).astype(np.float32)
                cam_rgb = np.asarray(cam_pcd.colors)*255.
                cam_rgb = cam_rgb.astype(np.uint8)
                cam_label = np.zeros((cam_rgb.shape[0], ), dtype=np.int32)
                sub_pts, sub_rgb, sub_lbls = grid_subsampling(cam_pts.astype(np.float32),
                                                            features=cam_rgb.astype(np.float32),
                                                            labels=cam_label.astype(np.int32),
                                                            sampleDl=0.04)
                
                # transform back for next frame
                hd_scene_pcd.transform(pose)
                    
                # compute mean
                # points and centers are all in world coordinate system
                sub_pts = (pose[:3, :3] @ sub_pts.T).T + pose[:3, 3]
                sub_cnt = np.mean(sub_pts, axis=0)
                # print(sub_pts.shape, sub_cnt)

                # store the center and point cloud
                scene_pcds.append(sub_pts)
                scene_ctrs.append(sub_cnt)
                scene_fids.append(i)
            # all_scene_frame_ids[scene] = scene_fids
            printProgressBar(nFrames, nFrames, prefix=task+' - '+scene, suffix='Complete', length=50, printEnd='\n')
            print(len(scene_pcds), len(scene_ctrs), len(scene_fids))
            # print(all_scene_frame_ids[scene])

            print('finished in', (time.time() - t)/60, 'mins')
            t = time.time()

            ## process the pcds to get positive and negative indices
            # initialise container
            one_scene_pos_neg = []
            for query in scene_ctrs:
                one_scene_pos_neg.append( ([], []) )
            # store the pos, neg indices
            for i, query in enumerate(scene_ctrs):
                # calculating overlaps takes forever. Using distance instead.
                # nnTree = KDTree(scene_pcds[i])
                
                for j, support in enumerate(scene_ctrs):
                    if j <= i:
                        continue
                    
                    else:
                        # pre-filtering by distance
                        dist = np.linalg.norm(query - support)

                        if dist > thre_dist:
                            # confirmed negative
                            one_scene_pos_neg[i][1].append( scene_fids[j] )
                            one_scene_pos_neg[j][1].append( scene_fids[i] )
                            # pcd_neg.append( scene_fids[j] )

                        else:
                            # # double check with overlap ratios
                            # nnDists, _ = nnTree.query(scene_pcds[j])
                            # overlap_num = np.sum(nnDists < 0.04)
                            # orQuery = overlap_num/scene_pcds[i].shape[0]
                            # orSupport = overlap_num/scene_pcds[j].shape[0]
                            # if orQuery > thre_ovlp and orSupport > thre_ovlp:
                            
                            # confirmed positive
                            one_scene_pos_neg[i][0].append( scene_fids[j] )
                            one_scene_pos_neg[j][0].append( scene_fids[i] )

                            #     # pcd_pos.append( scene_fids[j] )
                            #     # print(scene_fids[i], scene_fids[j], orQuery, orSupport)
                            # else:
                            #     # confirmed negative
                            #     one_scene_pos_neg[i][1].append( scene_fids[j] )
                            #     one_scene_pos_neg[j][1].append( scene_fids[i] )
                            #     # pcd_neg.append( scene_fids[j] )

                # one_scene_pos_neg.append( (pcd_pos, pcd_neg) )
                # print(i, ':', len(one_scene_pos_neg[i][0]), len(one_scene_pos_neg[i][1]))

            print('TEST:', 'total', len(one_scene_pos_neg), 'pos', len(one_scene_pos_neg[0][0]), 'neg', len(one_scene_pos_neg[0][1]))
            print('Get pos/neg indices takes', (time.time() - t)/60, 'mins.')
            all_scene_pos_neg[scene] = one_scene_pos_neg
            # # break a scene
            # break
        
        # NOTE positive indices and negative indices are not sorted!
        with open(os.path.join(scannet_path, 'vlad_'+task+'.txt'), 'wb') as vlad_file:
            pickle.dump(all_scene_pos_neg, vlad_file)
        print('vlad saved to', os.path.join(scannet_path, 'vlad_'+task+'.txt'))
        # # break a task
        # break

    # test
    with open(os.path.join(scannet_path, 'vlad_training.txt'), 'rb') as f:
        scenes_pn = pickle.load(f)

    print(len(scenes_pn))
    one_scene_pos_neg = scenes_pn['scene0000_00']
    print('TEST:', 'total', len(one_scene_pos_neg), 'pos', len(one_scene_pos_neg[0][0]), 'neg', len(one_scene_pos_neg[0][1]))
    

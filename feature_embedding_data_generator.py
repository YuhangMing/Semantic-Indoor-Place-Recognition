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
import random
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
    ### PCDs in camera coordinate frame
    # Set root paths
    scannet_path = '/media/yohann/Datasets/datasets/ScanNet'
    data_split_path = os.path.join(scannet_path, 'tools/Tasks/Benchmark')
    scannet_data_path = '/media/yohann/Datasets/datasets/ScanNet/scans'
    tasks = ['training', 'validation', 'test']
    
    # get non_zero_meaned pcd path and zero_meaned pcd path 
    pcd_prev_saved_path = os.path.join(scannet_data_path, 'input_pcd')
    if not os.path.exists(pcd_prev_saved_path):
        os.makedirs(pcd_prev_saved_path)
    pcd_new_path = os.path.join(scannet_data_path, 'input_pcd_0mean')
    if not os.path.exists(pcd_new_path):
        os.makedirs(pcd_new_path)
    
    # data holder
    all_files = {}          # dict, scene_name: [list of valid point cloud]
    all_scene_pos_neg = {}  # dict, scene_name: [list of ( pair of [list of pos pcd index], [list of neg pcd index])]
    # thresholds to get spatially sparser point clouds
    new_pcd_thres = 0.7
    new_cam_thres = 0.7
    # thresholds for get postive and negative point clouds
    pos_thre_dist = 2
    neg_thre_dist = 4
    test_threshold = 3

    # Loop through all tasks
    for task in tasks:
        print(task)

        # get scene names to process
        if task == 'training':
            scene_file_name = os.path.join(data_split_path, 'scannetv2_train.txt')
            scenes = np.sort(np.loadtxt(scene_file_name, dtype=str))
        elif task == 'validation':
            scene_file_name = os.path.join(data_split_path, 'scannetv2_val.txt')
            scenes = np.sort(np.loadtxt(scene_file_name, dtype=str))
        else:
            scene_file_name = os.path.join(data_split_path, 'scannetv2_test.txt')
            scenes = np.loadtxt(scene_file_name, dtype=str)
        
        # loop through the scenes
        t0 = time.time()
        pcd_count = 0   # count the total num of pcds in this TASK
        nScenes = len(scenes)
        for j, scene in enumerate(scenes):
            # # in validation, only keep one sequence per scene for better evaluation
            # if task == 'validation' and not '_00' in scene:
            #     continue

            # path to original scannet data
            scannet_scene_path = os.path.join(scannet_data_path, scene)
            # path to previously stored submap, non-zero-meaned and at 2 FPS
            scene_pcd_stored_path = os.path.join(pcd_prev_saved_path, scene)
            # path to store new submaps, spatially sparser 
            scene_pcd_new_path = os.path.join(pcd_new_path, scene)
            if not os.path.exists(scene_pcd_new_path):
                os.makedirs(scene_pcd_new_path)

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
            # ## Directly load submaps instead of cropping from the full map
            # # load the high definition point cloud
            # hd_scene_pcd = o3d.io.read_point_cloud(os.path.join(scannet_scene_path, scene+'_vh_clean.ply'))
            # # get intrinsics into matrix
            # intrinsics = o3d.camera.PinholeCameraIntrinsic(int(K[1]), int(K[0]), K[2], K[3], K[4], K[5])

            t = time.time()
            # loop through the frames to get pcd and cntr
            stepFrm = 15
            frame_count = 0
            scene_ctrs = []
            scene_pcds = []
            scene_fids = []
            scene_files = []
            prev_cntr = np.array([0,0,0])
            prev_pose = np.identity(4)
            for i in range(0, nFrames, stepFrm):
                printProgressBar(i+1, nFrames, prefix=scene+' '+str(j)+'/'+str(nScenes), suffix='Complete', length=50)
                file_name = scene+'_'+str(i)+'_sub.ply'
                
                # check if pose is lost
                pose = np.loadtxt(os.path.join(scannet_scene_path, 'pose', str(i)+'.txt'))
                chk_val = np.sum(pose)
                if np.isinf(chk_val) or np.isnan(chk_val):
                    continue

                # get subsampled point cloud
                sub_pcd_file = os.path.join(scene_pcd_stored_path, file_name)
                # print('reading from', sub_pcd_file)
                data = read_ply(sub_pcd_file)
                sub_pts = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T # Nx3
                if sub_pts.shape[0] < 2:
                    raise ValueError("Empty Polygan Mesh !!!!")
                sub_rgb = np.vstack((data['red'], data['green'], data['blue'])).astype(np.float32).T
                sub_lbls = data['class'].astype(np.int32)  # zeros for test set
                # Get center of the first frame in camera coordinates
                sub_cnt = np.mean(sub_pts, axis=0)

                # zero_mean
                sub_pts = sub_pts - sub_cnt
                # transform center to world coordinate system
                sub_cnt = pose[:3, :3] @ sub_cnt + pose[:3, 3]
            
                # check whether store the center and point cloud
                if len(scene_ctrs) == 0:
                    save_pcd_to_file = True
                    prev_cntr = sub_cnt
                    prev_pose = pose
                else:
                    dist_pcd = np.linalg.norm(sub_cnt - prev_cntr)
                    dist_cam = np.linalg.norm(pose[:3, 3] - prev_pose[:3, 3])
                    if dist_pcd > new_pcd_thres or dist_cam > new_cam_thres:
                        save_pcd_to_file = True
                        prev_cntr = sub_cnt
                        prev_pose = pose
                    else:
                        save_pcd_to_file = False
                if save_pcd_to_file:
                    scene_ctrs.append(sub_cnt)
                    # scene_fids.append(i)
                    scene_fids.append(frame_count)
                    scene_files.append(file_name)
                    new_sub_pcd_file = os.path.join(scene_pcd_new_path, file_name)
                    # print('saving to', new_sub_pcd_file, frame_count, pcd_count)
                    write_ply(new_sub_pcd_file,
                                (sub_pts.astype(np.float32), sub_rgb.astype(np.uint8), sub_lbls.astype(np.int32)), 
                                ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
                    frame_count += 1
                    pcd_count += 1

            # all_scene_frame_ids[scene] = scene_fids
            printProgressBar(nFrames, nFrames, prefix=scene+' '+str(j)+'/'+str(nScenes), suffix='Complete', length=50, printEnd='\n')
            print('- Get pcd information takes', (time.time() - t), 's')
            print('  debug display:',  len(scene_pcds), 'pcds,', len(scene_ctrs), 'centers,', len(scene_fids), 'frames, and max frame id is', frame_count)
            all_files[scene] = scene_files

            ## process the pcds to get positive and negative indices
            t = time.time()
            # initialise container
            one_scene_pos_neg = []  # ordered by index of frame, not actual frame id
            for query in scene_ctrs:
                one_scene_pos_neg.append( ([], []) )
            # store the pos, neg indices
            for i, query in enumerate(scene_ctrs):
                for j, support in enumerate(scene_ctrs):
                    if j <= i:
                        continue
                    else:
                        # computing distance distance
                        dist = np.linalg.norm(query - support)
                        # Negative samples
                        if dist > neg_thre_dist:
                            one_scene_pos_neg[i][1].append( scene_fids[j] )
                            one_scene_pos_neg[j][1].append( scene_fids[i] )
                        # Positive samples
                        if dist < pos_thre_dist:
                            one_scene_pos_neg[i][0].append( scene_fids[j] )
                            one_scene_pos_neg[j][0].append( scene_fids[i] )

            print('- Get pos/neg indices takes', (time.time() - t), 's.')
            print('  eg. display:', 'total', len(one_scene_pos_neg), 'pos', len(one_scene_pos_neg[0][0]), 'neg', len(one_scene_pos_neg[0][1]))

            all_scene_pos_neg[scene] = one_scene_pos_neg
            # # break a scene
            # break
        print(task, ':', pcd_count, 'pcds stored, and in total takes', (time.time() - t0), 's')
        # # break a task
        # break

    print(len(all_files), len(all_scene_pos_neg))    
    # NOTE positive indices and negative indices are not sorted!
    with open(os.path.join(scannet_path, 'vlad_pcd.txt'), 'wb') as pcd_file:
        pickle.dump(all_files, pcd_file)
    print('\nPCD saved to', os.path.join(scannet_path, 'pcd_list.txt'))
    with open(os.path.join(scannet_path, 'vlad_pos_neg.txt'), 'wb') as vlad_file:
        pickle.dump(all_scene_pos_neg, vlad_file)
    print('VLAD saved to', os.path.join(scannet_path, 'vlad_pos_neg.txt\n'))


    #### display test
    print('Test with loading')
    with open(os.path.join(scannet_path, 'vlad_pos_neg.txt'), 'rb') as f:
        scenes_pn = pickle.load(f)
    key = random.choice(list(scenes_pn.keys()))
    # key = 'scene0000_00'
    one_scene_pos_neg = scenes_pn[key]
    print(key, 'from', len(scenes_pn), 'scenes in total.')
    print('eg pos: ', len(one_scene_pos_neg[0][0]), one_scene_pos_neg[0][0])
    print('eg neg: ', len(one_scene_pos_neg[0][1]), one_scene_pos_neg[0][1])
    with open(os.path.join(scannet_path, 'vlad_pcd.txt'), 'rb') as f:
        scenes_pcd = pickle.load(f)
    one_scene_pcd = scenes_pcd[key]
    print('# of pcd:', len(one_scene_pos_neg), len(one_scene_pcd))
    print(one_scene_pcd)
    
    # ###############
    # # get file names list from stored files
    # path_to_pcds = os.path.join(scannet_data_path, 'input_new_pcd')
    # scene_list = os.listdir(path_to_pcds)
    # nScenes = len(scene_list)
    # scene_list.sort()
    # files = {}
    # uniform_length = 13+4+8
    # for sId, scene in enumerate(scene_list):
    #     # printProgressBar(sId+1, nScenes, prefix='PCDs', suffix='Complete', length=50)

    #     scene_path = os.path.join(path_to_pcds, scene)
    #     pcd_list = os.listdir(scene_path)
    #     pcd_uni_len = os.listdir(scene_path)
        
    #     # get sorted index
    #     for i, pcd in enumerate(pcd_uni_len):
    #         diff = uniform_length - len(pcd)
    #         pcd_uni_len[i] = pcd[:13]+'0'*diff+pcd[13:]
    #     pcd_uni_len_np = np.array(pcd_uni_len)
    #     sorted_idx = np.argsort(pcd_uni_len_np)
        
    #     # sort the input file list
    #     scene_files = []
    #     for i in sorted_idx:
    #         scene_files.append(pcd_list[i])
    #     files[scene] = scene_files

    #     # # TEST
    #     # for file in scene_files:
    #     #     print(file[13:-8])
    #     # break

    # with open(os.path.join(scannet_path, 'pcd_list.txt'), 'wb') as pcd_file:
    #     pickle.dump(files, pcd_file)
    
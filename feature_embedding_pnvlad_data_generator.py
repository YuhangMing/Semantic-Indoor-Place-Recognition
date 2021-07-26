import os
import shutil
import pickle
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from os import listdir, makedirs
from os.path import join, exists
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
    #### voxel grid downsample ####
    ## Generate PointNetVLAD input data
    # get path info
    data_path = '/media/yohann/Datasets/datasets/ScanNet'
    #### Generate pointnetvlad input data
    data_split_path = join(data_path, 'tools/Tasks/Benchmark')
    # 4096, 0 meaned, camera coordinate frame
    pnvlad_pcd_path = join(data_path, 'scans', 'pnvlad_pcd')
    if not exists(pnvlad_pcd_path):
        makedirs(pnvlad_pcd_path)

    # Load pre-processed pos-neg indices AND file names
    # all tasks are stored in a single file
    vlad_pn_file = join(data_path, 'VLAD_triplets', 'vlad_pos_neg.txt')
    with open(vlad_pn_file, "rb") as f:
        # dict, key = scene string, val = list of pairs of (list pos, list neg)
        all_scene_pos_neg = pickle.load(f)
    valid_pcd_file = join(data_path, 'VLAD_triplets', 'vlad_pcd.txt')
    with open(valid_pcd_file, "rb") as f:
        # dict, key = scene string, val = list of filenames
        all_scene_pcds = pickle.load(f)

    tasks = ['training', 'validation', 'test']
    invalid = []
    dls_over = [0.02, 0.01, 0.005, 0.0025, 0.00125]
    dls_less = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    for task in tasks:
        print(task)

        # get scene names to process
        if task == 'training':
            scene_file_name = join(data_split_path, 'scannetv2_train.txt')
            scenes = np.sort(np.loadtxt(scene_file_name, dtype=str))
        elif task == 'validation':
            scene_file_name = join(data_split_path, 'scannetv2_val.txt')
            scenes = np.sort(np.loadtxt(scene_file_name, dtype=str))
        else:
            scene_file_name = join(data_split_path, 'scannetv2_test.txt')
            scenes = np.loadtxt(scene_file_name, dtype=str)

        for scene_idx, scene_name in enumerate(scenes):
            print(scene_name)
            # if scene_name != 'scene0014_00':
            #     continue
            data_scene_path = join(data_path, 'scans', scene_name)
            # path to store new submaps, 4096 pts, in binary format 
            pnvlad_pcd_scene_path = join(pnvlad_pcd_path, scene_name)
            if not exists(pnvlad_pcd_scene_path):
                makedirs(pnvlad_pcd_scene_path)

            # get num of frames and the intrinsic matrix
            K = []
            nFrames = 0
            info_file = open(join(data_scene_path, scene_name+'.txt'))
            for line in info_file:
                vals = line.strip().split(' ')

                if 'depth' in vals[0]:
                    K.append(float(vals[2]))
                if vals[0] == 'numDepthFrames':
                    nFrames = int(vals[2])

            # load the high definition point cloud
            hd_scene_pcd = o3d.io.read_point_cloud(join(
                    data_scene_path, scene_name+'_vh_clean.ply'
                ))
            # get intrinsics into matrix
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                int(K[1]), int(K[0]), K[2], K[3], K[4], K[5]
            )

            # get pcd file names for current scene
            scene_pcd_names = all_scene_pcds[scene_name]
            for pcd_idx, pcd_filename in enumerate(scene_pcd_names):
                actual_frame_id = int(pcd_filename[13:-8])
                # if actual_frame_id != 2235:
                #     continue
                # get frame pose
                pose = np.loadtxt(join(data_scene_path, 'pose', str(actual_frame_id)+'.txt'))
                chk_val = np.sum(pose)
                if np.isinf(chk_val) or np.isnan(chk_val):
                    raise ValueError('Invalid pose value.')
                
                # back project the raw depth 
                depth = o3d.io.read_image(join(
                    data_scene_path, 'depth', str(actual_frame_id)+'.png'))
                depth_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics)

                # get aligned bounding box
                bbox_aligned = depth_pcd.get_axis_aligned_bounding_box()

                # crop pcd of interest
                cam_pcd = hd_scene_pcd.transform(np.linalg.inv(pose)).crop(bbox_aligned)

                # save the sub-pcd in the original resolution
                cam_pts = np.asarray(cam_pcd.points).astype(np.float32)
                cam_rgb = np.asarray(cam_pcd.colors)*255.
                cam_rgb = cam_rgb.astype(np.uint8)
                cam_label = np.zeros((cam_rgb.shape[0], ), dtype=np.int32)

                num_pts = cam_pts.shape[0]
                if num_pts < 4096:
                    invalid.append((scene_name, pcd_filename))
                    continue
                
                # voxel grid downsample the point cloud
                sub_pts, sub_rgb, sub_lbls = grid_subsampling(cam_pts.astype(np.float32),
                                                            features=cam_rgb.astype(np.float32),
                                                            labels=cam_label.astype(np.int32),
                                                            sampleDl=0.04)
                
                num_subpts = sub_pts.shape[0]
                # print(num_pts, num_subpts)
                if num_subpts < 4096:
                    scale = int(4096/num_subpts)+1
                    count = 0
                    while scale > 1:
                        if count >= len(dls_less):
                            break
                        sub_pts, sub_rgb, sub_lbls = grid_subsampling(cam_pts.astype(np.float32),
                                                                features=cam_rgb.astype(np.float32),
                                                                labels=cam_label.astype(np.int32),
                                                                sampleDl=dls_over[count])
                        count += 1
                        num_subpts = sub_pts.shape[0]
                        scale = int(4096/num_subpts)+1
                
                else:
                    scale = int(num_subpts/4096)-1
                    count = 0
                    while scale > 1:
                        if count >= len(dls_less):
                            break
                        sub_pts, sub_rgb, sub_lbls = grid_subsampling(cam_pts.astype(np.float32),
                                                            features=cam_rgb.astype(np.float32),
                                                            labels=cam_label.astype(np.int32),
                                                            sampleDl=dls_less[count])
                        count += 1
                        num_subpts = sub_pts.shape[0]
                        scale = int(num_subpts/4096)-1
                        # print(dls_less[count], sub_pts.shape, scale)

                num_subpts = sub_pts.shape[0]
                # print(scale, num_subpts)
                if num_subpts > 4096:
                    input_inds = np.random.choice(num_subpts, size=4096, replace=False)
                    sub_pts = sub_pts[input_inds, :]
                else:
                    print(scene_name, pcd_filename, '< 4096')
                
                # store the subpcd in binary file
                # print(sub_pts.shape)
                bin_file = join(pnvlad_pcd_scene_path, pcd_filename[:-3]+'bin')
                sub_pts.astype(np.float64).tofile(bin_file)

                # transform back for next frame
                hd_scene_pcd.transform(pose)

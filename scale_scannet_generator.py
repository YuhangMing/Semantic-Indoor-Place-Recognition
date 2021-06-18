#
#
#      0=================================0
#      |    Store (extra) small pcd      |
#      |    from original sequences      |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Generate significant scale changed point clouds
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Yuhang Ming
#

import os
import numpy as np
import open3d as o3d

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

def print_bbox_info(scene, frameId, bbox):
    print(scene, str(frameId)+'.png')
    print(' - volume =', bbox.volume())
    # print(' - max_bd =', bbox.max_bound)
    # print(' - min_bd =', bbox.min_bound)
    print(' - diffs =', bbox.max_bound - bbox.min_bound)
    # print()


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


        # start new file
        small_regions_file = open(os.path.join(scannet_path, 'small_regions_'+task+'.txt'), 'w')
        small_regions_file.close()
        extreme_small_file = open(os.path.join(scannet_path, 'small_regions_extreme_'+task+'.txt'), 'w')
        extreme_small_file.close()
        for scene in scenes:
            scannet_scene_path = os.path.join(scannet_data_path, scene)
            scannet_pcd_path = os.path.join(scannet_data_path, 'input_pcd', scene)
            # print(scannet_scene_path)
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
            # print(nFrames)

            # load the high definition point cloud
            hd_scene_pcd = o3d.io.read_point_cloud(os.path.join(
                    scannet_scene_path, scene+'_vh_clean.ply'
                ))
            # get intrinsics into matrix
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                int(K[1]), int(K[0]), K[2], K[3], K[4], K[5]
            )

            # loop through the frames
            stepFrm = 30
            frm_centers = []
            for i in range(0, nFrames, stepFrm):
                bSave = True
                printProgressBar(i+1, nFrames, prefix=task+' - '+scene, suffix='Complete', length=50)
                frame_pcd_file = os.path.join(scannet_pcd_path, scene+'_'+str(i)+'.ply')
                frame_subpcd_file = os.path.join(scannet_pcd_path, scene+'_'+str(i)+'_sub.ply')
                pose = np.loadtxt(os.path.join(scannet_scene_path, 'pose', str(i)+'.txt'))
                # check if pose is lost
                chk_val = np.sum(pose)
                if np.isinf(chk_val) or np.isnan(chk_val):
                    # print('Invalid pose for', scene, 'frame', i)
                    continue
                if os.path.exists(frame_pcd_file) and os.path.exists(frame_subpcd_file):
                    bSave = False
                    # continue

                # back project the raw depth 
                depth = o3d.io.read_image(os.path.join(
                    scannet_scene_path, 'depth', str(i)+'.png'))
                depth_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics)

                # get aligned bounding box
                bbox_aligned = depth_pcd.get_axis_aligned_bounding_box()
                bbox_volume = bbox_aligned.volume()
                bbox_xyz_diff = bbox_aligned.max_bound - bbox_aligned.min_bound
                
                # volume threshold and dimension threshold
                if bbox_volume < 3.5 or min(bbox_xyz_diff) < 0.5:
                    if bSave:
                        # crop pcd of interest
                        cam_pcd = hd_scene_pcd.transform(np.linalg.inv(pose)).crop(bbox_aligned)

                        # save the sub-pcd in the original resolution
                        cam_pts = np.asarray(cam_pcd.points).astype(np.float32)
                        cam_rgb = np.asarray(cam_pcd.colors)*255.
                        cam_rgb = cam_rgb.astype(np.uint8)
                        cam_label = np.zeros((cam_rgb.shape[0], ), dtype=np.int32)

                        # save as ply
                        write_ply(frame_pcd_file,
                                (cam_pts, cam_rgb, cam_label), 
                                ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

                        # voxel grid downsample the point cloud
                        sub_pts, sub_rgb, sub_lbls = grid_subsampling(cam_pts.astype(np.float32),
                                                                    features=cam_rgb.astype(np.float32),
                                                                    labels=cam_label.astype(np.int32),
                                                                    sampleDl=0.04)

                        # save as ply
                        write_ply(frame_subpcd_file,
                                (sub_pts.astype(np.float32), sub_rgb.astype(np.uint8), sub_lbls.astype(np.int32)), 
                                ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

                        # transform back for next frame
                        hd_scene_pcd.transform(pose)
                    else:
                        data  = read_ply(frame_subpcd_file)
                        sub_pts = np.vstack((data['x'], data['y'], data['z'])).T # Nx3
                    
                    # compute mean
                    current_center = np.mean(sub_pts, axis=0)
                    current_center = pose[:3, :3] @ current_center + pose[:3, 3]
                    # print(current_center)
                    bAddToDB = True
                    for db_center in frm_centers:
                        dist = np.linalg.norm(current_center - db_center)
                        if dist < 1.0:
                            bAddToDB = False
                            break

                    # store current pcd information
                    if bAddToDB:
                        frm_centers.append(current_center)
                        with open(os.path.join(scannet_path, 'small_regions_'+task+'.txt'), 'a') as small_file:
                            small_file.write(
                                frame_subpcd_file.split('/')[-1]
                                + ' ' + str(bbox_volume)
                                + ' ' + str(bbox_xyz_diff[0])
                                + ' ' + str(bbox_xyz_diff[1])
                                + ' ' + str(bbox_xyz_diff[2])
                                +'\n'
                        )
                        if bbox_aligned.volume() < 1.0:
                            with open(os.path.join(scannet_path, 'small_regions_extreme_'+task+'.txt'), 'a') as extreme_file:
                                extreme_file.write(frame_subpcd_file.split('/')[-1]+'\n')
                    continue
            printProgressBar(nFrames, nFrames, prefix=task+' - '+scene, suffix='Complete', length=50, printEnd='\n')



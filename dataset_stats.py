#
#      Performing Dataset Statistics Analysis
#      Yuhang Ming
#


import pickle
import numpy as np
from os import makedirs
from os.path import join, exists, isfile

from utils.ply import read_ply

if __name__ == '__main__':
    """
    Number of pts:
    task        <5000		5000-6000	6000-7000	7000-8000	8000-9000	>9000
    training    [6089		2383		2409		2467		2459		19295]
                [0.1734659	0.06788787	0.06862857	0.0702809	0.07005299	0.54968378]
    validation  [1710		655         615		    608		    588		    5517]
                [0.17641597	0.06757454	0.06344785	0.06272568	0.06066233	0.56917363]
    test        [629		224		    207		    208		    214		    2126]
                [0.17433481	0.06208426	0.05737251	0.05764967	0.05931264	0.58924612]
    """

    # set dir
    base_dir = '/media/yohann/fastStorage/data/'
    scannet_path = '/media/yohann/fastStorage/data/ScanNetPR'
    data_split_path = join(scannet_path, 'tools/Tasks/Benchmark')
    scannet_data_path = join(scannet_path, 'scans')
    tasks = ['training', 'validation', 'test']

    # load all valid pcds for all tasks
    valid_pcd_file = join(scannet_path, 'VLAD_triplets', 'vlad_pcd.txt')
    with open(valid_pcd_file, "rb") as f:
        # dict: key = scene string, val = list of filenames
        all_scene_pcds = pickle.load(f)

    ########################
    ## Statistic analysis ##
    ########################
    stats = []
    num = {'training': 35102, 'validation': 9693, 'test': 3608}
    count = []

    dict_pcd_size_ind = {}
    dict_pcd_size = {}
    # Loop through all tasks
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
        # print(' ', str(len(scenes)), 'sequences')

        task_pcd_size_ind = {'<5000': [], '5000-10000': [], '10000-15000': [], '>15000': []}
        task_stats = np.array([0,0,0,0])
        task_count = 0
        # loop through all scenes
        for i, scene in enumerate(scenes):
            num_scene_pcd = len(all_scene_pcds[scene])
            
            scene_pcd_size = []
            # Loop through pcds
            for j, pcd in enumerate(all_scene_pcds[scene]):
                print('{}%'.format(int(100*task_count/num[task])), flush=True, end='\r')
                # # Get frame id 
                # actual_frame_id = int(pcd[13:-8])
                
                # Get pcd file path
                pcd_file_path = join(scannet_data_path, scene, 'input_pcd_0mean', pcd)

                # load file
                data = read_ply(pcd_file_path)
                pts = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T
                num_pts = pts.shape[0]

                scene_pcd_size.append(num_pts)

                if num_pts < 5000:
                    task_stats[0] += 1
                    task_pcd_size_ind['<5000'].append((i,j))
                elif num_pts < 10000:
                    task_stats[1] += 1
                    task_pcd_size_ind['5000-10000'].append((i,j))
                elif num_pts < 15000:
                    task_stats[2] += 1
                    task_pcd_size_ind['10000-15000'].append((i,j))
                else:
                    task_stats[3] += 1
                    task_pcd_size_ind['>15000'].append((i,j))
                task_count += 1

            dict_pcd_size[scene] = scene_pcd_size

        dict_pcd_size_ind[task] = task_pcd_size_ind
        print('100%')
        stats.append(task_stats)
        count.append(task_count)
        print(task_stats)
        print(task_stats/task_count)

    print(len(dict_pcd_size_ind['training']['<5000']))
    print(len(dict_pcd_size_ind['validation']['<5000']))
    print(len(dict_pcd_size_ind['test']['<5000']))
    # print(dict_pcd_size['scene0000_00'])

    # pcd_size_file = join(scannet_path, 'VLAD_triplets', 'pcd_size.txt')
    # with open(pcd_size_file, "wb") as f:
    #     pickle.dump(dict_pcd_size, f)
    
    # pcd_size_ind_file = join(scannet_path, 'VLAD_triplets', 'pcd_size_ind.txt')
    # with open(pcd_size_ind_file, "wb") as f:
    #     pickle.dump(dict_pcd_size_ind, f)
    
    print(stats)
    
    # ##########################
    # ## Dataset manipulation ##
    # ##########################
    # import shutil
    # scannet_pr_path = join(base_dir, 'ScanNetPR')
    # if not exists(scannet_pr_path):
    #     makedirs(scannet_pr_path)
    # pcd_path = join(scannet_pr_path, 'input_pcd')
    # # Retrive All relevant pcds and RGB-D images and store it in a new directory
    # # Loop though all PR pcds
    # count = 0
    # max_count = len(all_scene_pcds)
    # for scene, scene_pcds in all_scene_pcds.items():
    #     # print(scene, '\n')
    #     # print('{}%'.format(int(100*count/max_count)), flush=True, end='\r')
    #     # new scene folder
    #     new_scene_path = join(scannet_pr_path, scene)
    #     # print(new_scene_path)
    #     if not exists(new_scene_path):
    #         makedirs(new_scene_path)
        
    #     # loop through all pcds in the scene
    #     # store pcds, rgb, depth, pose, intrinsics in corresponding folder 
    #     pre_input_pcd_path = join(scannet_data_path, 'input_pcd', scene)
    #     new_input_pcd_path = join(new_scene_path, 'input_pcd')
    #     if not exists(new_input_pcd_path):
    #         makedirs(new_input_pcd_path)
        
    #     pre_input_pcd_0mean_path = join(scannet_data_path, 'input_pcd_0mean', scene)
    #     new_input_pcd_0mean_path = join(new_scene_path, 'input_pcd_0mean')
    #     if not exists(new_input_pcd_0mean_path):
    #         makedirs(new_input_pcd_0mean_path)
        
    #     pre_pnvlad_pcd_path = join(scannet_data_path, 'pnvlad_pcd', scene)
    #     new_pnvlad_pcd_path = join(new_scene_path, 'pnvlad_pcd')
    #     if not exists(new_pnvlad_pcd_path):
    #         makedirs(new_pnvlad_pcd_path)
        
    #     pre_color_path = join(scannet_data_path, scene, 'color')
    #     new_color_path = join(new_scene_path, 'color')
    #     if not exists(new_color_path):
    #         makedirs(new_color_path)
        
    #     pre_depth_path = join(scannet_data_path, scene, 'depth')
    #     new_depth_path = join(new_scene_path, 'depth')
    #     if not exists(new_depth_path):
    #         makedirs(new_depth_path)
        
    #     pre_pose_path = join(scannet_data_path, scene, 'pose')
    #     new_pose_path = join(new_scene_path, 'pose')
    #     if not exists(new_pose_path):
    #         makedirs(new_pose_path)
        
    #     pre_info_file = join(scannet_data_path, scene, scene+'.txt')
    #     new_info_file = join(new_scene_path, scene+'.txt')
    #     if isfile(pre_info_file):
    #         shutil.copy(pre_info_file, new_info_file)
        
    #     max_pcd = len(scene_pcds)
    #     for j, pcd in enumerate(scene_pcds):
    #         actual_frame_id = str(int(pcd[13:-8]))
    #         print('{:.2f}%  - {:.2f}%'.format(float(100*count/max_count), float(100*j/max_pcd)), flush=True, end='\r')
    #         # print('  ', actual_frame_id, pcd)

    #         # get file dirs
    #         src_files = []
    #         dst_files = []
            
    #         # pcds
    #         src_files.append(join(pre_input_pcd_path, pcd))
    #         dst_files.append(join(new_input_pcd_path, pcd))

    #         src_files.append(join(pre_input_pcd_0mean_path, pcd))
    #         dst_files.append(join(new_input_pcd_0mean_path, pcd))
            
    #         src_files.append(join(pre_pnvlad_pcd_path, pcd[:-3]+'bin'))
    #         dst_files.append(join(new_pnvlad_pcd_path, pcd[:-3]+'bin'))
    #         # color, depth, pose
    #         src_files.append(join(pre_color_path, actual_frame_id+'.jpg'))
    #         dst_files.append(join(new_color_path, actual_frame_id+'.jpg'))

    #         src_files.append(join(pre_depth_path, actual_frame_id+'.png'))
    #         dst_files.append(join(new_depth_path, actual_frame_id+'.png'))
            
    #         src_files.append(join(pre_pose_path, actual_frame_id+'.txt'))
    #         dst_files.append(join(new_pose_path, actual_frame_id+'.txt'))

    #         # copy
    #         for i in range(len(src_files)):
    #             if isfile(src_files[i]):
    #                 shutil.copy(src_files[i], dst_files[i])
    #     count += 1
    #     # print('{:.2f}% - 100\%'.format(float(100*count/max_count)), flush=True, end='\r')

    # # # test pcd files
    # # data = read_ply('/media/adam/Datasets/datasets/ScanNetPR/scene0000_00/input_pcd/scene0000_00_0_sub.ply')
    # # pts = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T
    # # num_pts = pts.shape[0]
    # # mean = np.mean(pts, axis=0)
    # # print(mean)
    # # data = read_ply('/media/adam/Datasets/datasets/ScanNetPR/scene0000_00/input_pcd_0mean/scene0000_00_0_sub.ply')
    # # pts = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T
    # # num_pts = pts.shape[0]
    # # mean = np.mean(pts, axis=0)
    # # print(mean)
    
    

            

    # # Loop through all tasks
    # for task in tasks:
    #     print(task)

    #     # get scene names to process
    #     if task == 'training':
    #         scene_file_name = join(data_split_path, 'scannetv2_train.txt')
    #         scenes = np.sort(np.loadtxt(scene_file_name, dtype=str))
    #     elif task == 'validation':
    #         scene_file_name = join(data_split_path, 'scannetv2_val.txt')
    #         scenes = np.sort(np.loadtxt(scene_file_name, dtype=str))
    #     else:
    #         scene_file_name = join(data_split_path, 'scannetv2_test.txt')
    #         scenes = np.loadtxt(scene_file_name, dtype=str)
    #     # print(' ', str(len(scenes)), 'sequences')

    #     task_pcd_size_ind = {'<5000': [], '5000-7000': [], '7000-9000': [], '>9000': []}
    #     task_stats = np.array([0,0,0,0])
    #     task_count = 0
    #     # loop through all scenes
    #     for i, scene in enumerate(scenes):
    #         num_scene_pcd = len(all_scene_pcds[scene])
            
    #         scene_pcd_size = []
    #         # Loop through pcds
    #         for j, pcd in enumerate(all_scene_pcds[scene]):
    #             print('{}%'.format(int(100*task_count/num[task])), flush=True, end='\r')
    #             # # Get frame id 
    #             # actual_frame_id = int(pcd[13:-8])
                
    #             # Get pcd file path
    #             pcd_file_path = join(scannet_data_path, 'input_pcd_0mean', scene, pcd)


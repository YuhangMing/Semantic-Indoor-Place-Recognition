import pickle
import numpy as np
from os.path import join

from utils.ply import read_ply

if __name__ == '__main__':
    """
    Number of pts:
    task        <5000		5000-6000	6000-7000	7000-8000	8000-9000	>9000
    training    [6089		2383		2409		2467		2459		19295]
                [0.1734659	0.06788787	0.06862857	0.0702809	0.07005299	0.54968378]
    validation  [1710		655		615		608		588		5517]
                [0.17641597	0.06757454	0.06344785	0.06272568	0.06066233	0.56917363]
    test        [629		224		207		208		214		2126]
                [0.17433481	0.06208426	0.05737251	0.05764967	0.05931264	0.58924612]
    """

    # set dir
    scannet_path = '/media/adam/Datasets/datasets/ScanNet'
    data_split_path = join(scannet_path, 'tools/Tasks/Benchmark')
    scannet_data_path = join(scannet_path, 'scans')
    tasks = ['training', 'validation', 'test']

    # load all valid pcds for all tasks
    valid_pcd_file = join(scannet_path, 'VLAD_triplets', 'vlad_pcd.txt')
    with open(valid_pcd_file, "rb") as f:
        # dict: key = scene string, val = list of filenames
        all_scene_pcds = pickle.load(f)
    
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

        task_pcd_size_ind = {'<5000': [], '5000-7000': [], '7000-9000': [], '>9000': []}
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
                pcd_file_path = join(scannet_data_path, 'input_pcd_0mean', scene, pcd)

                # load file
                data = read_ply(pcd_file_path)
                pts = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T
                num_pts = pts.shape[0]

                scene_pcd_size.append(num_pts)

                # if num_pts < 5000:
                #     task_stats[0] += 1
                #     task_pcd_size_ind['<5000'].append((i,j))
                # elif num_pts < 7000:
                #     task_stats[1] += 1
                #     task_pcd_size_ind['5000-7000'].append((i,j))
                # elif num_pts < 9000:
                #     task_stats[2] += 1
                #     task_pcd_size_ind['7000-9000'].append((i,j))
                # else:
                #     task_stats[3] += 1
                #     task_pcd_size_ind['>9000'].append((i,j))
                task_count += 1

            dict_pcd_size[scene] = scene_pcd_size

    #     dict_pcd_size_ind[task] = task_pcd_size_ind
    #     print('100%')
    #     stats.append(task_stats)
    #     count.append(task_count)
    #     print(task_stats)
    #     print(task_stats/task_count)

    # print(len(dict_pcd_size_ind['training']['<5000']))
    # print(len(dict_pcd_size_ind['validation']['<5000']))
    # print(len(dict_pcd_size_ind['test']['<5000']))

    print(dict_pcd_size['scene0000_00'])

    pcd_size_file = join(scannet_path, 'VLAD_triplets', 'pcd_size.txt')
    with open(pcd_size_file, "wb") as f:
        pickle.dump(dict_pcd_size, f)
    
    # pcd_size_ind_file = join(scannet_path, 'VLAD_triplets', 'pcd_size_ind.txt')
    # with open(pcd_size_ind_file, "wb") as f:
    #     pickle.dump(dict_pcd_size_ind, f)
    
    # print(stats)


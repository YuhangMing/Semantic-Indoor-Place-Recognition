import os
import shutil
import pickle
import matplotlib.pyplot as plt
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

if __name__ == '__main__':
    # #### Plot Legends ####
    # dataset = 'ScanNet'
    # if dataset == 'S3DIS':
    #     label_to_names = {0: 'ceiling',
    #                     1: 'floor',
    #                     2: 'wall',
    #                     3: 'beam',
    #                     4: 'column',
    #                     5: 'window',
    #                     6: 'door',
    #                     7: 'chair',
    #                     8: 'table',
    #                     9: 'bookcase',
    #                     10: 'sofa',
    #                     11: 'board',
    #                     12: 'clutter'
    #                     }
    #     label_to_colour = {0: [ 233, 229, 107], #'ceiling' .-> .yellow
    #                     1: [  95, 156, 196], #'floor' .-> . blue
    #                     2: [ 179, 116,  81], #'wall'  ->  brown
    #                     3: [ 241, 149, 131], #'beam'  ->  salmon
    #                     4: [  81, 163, 148], #'column'  ->  bluegreen
    #                     5: [  77, 174,  84], #'window'  ->  bright green
    #                     6: [ 108, 135,  75], #'door'   ->  dark green
    #                     7: [  41,  49, 101], #'chair'  ->  darkblue
    #                     8: [  79,  79,  76], #'table'  ->  dark grey
    #                     9: [223,  52,  52], #'bookcase'  ->  red
    #                     10: [ 89,  47,  95], #'sofa'  ->  purple
    #                     11: [ 81, 109, 114], #'board'   ->  grey
    #                     12: [233, 233, 229], #'clutter'  ->  light grey
    #                     13: [0   ,   0,   0], #unlabelled .->. black
    #                     }
    # elif dataset == 'ScanNet':
    #     # subset of 20 classes from NYUv2's 40 classes
    #     label_to_names = {0: 'unclassified',
    #                         1: 'wall',
    #                         2: 'floor',
    #                         3: 'cabinet',
    #                         4: 'bed',
    #                         5: 'chair',
    #                         6: 'sofa',
    #                         7: 'table',
    #                         8: 'door',
    #                         9: 'window',
    #                         10: 'bookshelf',
    #                         11: 'picture',
    #                         12: 'counter',
    #                         14: 'desk',
    #                         16: 'curtain',
    #                         24: 'refridgerator',
    #                         28: 'shower curtain',
    #                         33: 'toilet',
    #                         34: 'sink',
    #                         36: 'bathtub',
    #                         39: 'other furniture'
    #                     }
    #     label_to_colour = {0: [0, 0, 0], # black -> 'unclassified'
    #                         1: [174, 198, 232], # light purple -> 'wall',
    #                         2: [151, 223, 137], # lime -> 'floor',
    #                         3: [31, 120, 180], # dark blue -> 'cabinet'
    #                         4: [255, 188, 120], # light orange -> 'bed',
    #                         5: [188, 189, 35], #  -> 'chair',
    #                         6: [140, 86, 74], # brown -> 'sofa',
    #                         7: [255, 152, 151], # pink -> 'table',
    #                         8: [213, 39, 40], # red -> 'door',
    #                         9: [196, 176, 213], # light purple -> 'window',
    #                         10: [150, 102, 188], # purple -> 'bookshelf',
    #                         11: [196, 156, 148], # light brown -> 'picture',
    #                         12: [23, 190, 208], # dark cyan -> 'counter',
    #                         14: [247, 183, 210], # light pink -> 'desk',
    #                         16: [218, 219, 141], #  -> 'curtain',
    #                         24: [254, 127, 14], # orange -> 'refridgerator',
    #                         28: [158, 218, 229], # light cyan -> 'shower curtain',
    #                         33: [43, 160, 45], # green -> 'toilet',
    #                         34: [112, 128, 144], # grey -> 'sink',
    #                         36: [227, 119, 194], #  -> 'bathtub',
    #                         39: [82, 83, 163], # dark purple -> 'otherfurniture'
    #                     }
    # elif dataset == 'SemanticKitti':
    #     label_to_names = { 0 : "unlabeled",
    #                         1 : "outlier",
    #                         10: "car",
    #                         11: "bicycle",
    #                         13: "bus",
    #                         15: "motorcycle",
    #                         16: "on-rails",
    #                         18: "truck",
    #                         20: "other-vehicle",
    #                         30: "person",
    #                         31: "bicyclist",
    #                         32: "motorcyclist",
    #                         40: "road",
    #                         44: "parking",
    #                         48: "sidewalk",
    #                         49: "other-ground",
    #                         50: "building",
    #                         51: "fence",
    #                         52: "other-structure",
    #                         60: "lane-marking",
    #                         70: "vegetation",
    #                         71: "trunk",
    #                         72: "terrain",
    #                         80: "pole",
    #                         81: "traffic-sign",
    #                         99: "other-object",
    #                         252: "moving-car",
    #                         253: "moving-bicyclist",
    #                         254: "moving-person",
    #                         255: "moving-motorcyclist",
    #                         256: "moving-on-rails",
    #                         257: "moving-bus",
    #                         258: "moving-truck",
    #                         259: "moving-other-vehicle"
    #     }
    #     label_map_inv = {0: 0,      # "unlabeled", and others ignored
    #                     1: 10,     # "car"
    #                     2: 11,     # "bicycle"
    #                     3: 15,    # "motorcycle"
    #                     4: 18,     # "truck"
    #                     5: 20,     # "other-vehicle"
    #                     6: 30,     # "person"
    #                     7: 31,     # "bicyclist"
    #                     8: 32,     # "motorcyclist"
    #                     9: 40,     # "road"
    #                     10: 44,    # "parking"
    #                     11: 48,    # "sidewalk"
    #                     12: 49,    # "other-ground"
    #                     13: 50,    # "building"
    #                     14: 51,    # "fence"
    #                     15: 70,    # "vegetation"
    #                     16: 71,    # "trunk"
    #                     17: 72,    # "terrain"
    #                     18: 80,    # "pole"
    #                     19: 81,    # "traffic-sign"
    #                     20: 252,    # "moving-car"
    #                     21: 253,    # "moving-bicyclist"
    #                     22: 254,    # "moving-person"
    #                     23: 255,    # "moving-motorcyclist"
    #                     24: 259,    # "moving-other-vehicle"
    #                     25: 258,    # "moving-truck"
    #     }
    #     label_to_colour = {0 : [0, 0, 0],
    #                         1 : [0, 0, 255],
    #                         10: [245, 150, 100],
    #                         11: [245, 230, 100],
    #                         13: [250, 80, 100],
    #                         15: [150, 60, 30],
    #                         16: [255, 0, 0],
    #                         18: [180, 30, 80],
    #                         20: [255, 0, 0],
    #                         30: [30, 30, 255],
    #                         31: [200, 40, 255],
    #                         32: [90, 30, 150],
    #                         40: [255, 0, 255],
    #                         44: [255, 150, 255],
    #                         48: [75, 0, 75],
    #                         49: [75, 0, 175],
    #                         50: [0, 200, 255],
    #                         51: [50, 120, 255],
    #                         52: [0, 150, 255],
    #                         60: [170, 255, 150],
    #                         70: [0, 175, 0],
    #                         71: [0, 60, 135],
    #                         72: [80, 240, 150],
    #                         80: [150, 240, 255],
    #                         81: [0, 0, 255],
    #                         99: [255, 255, 50],
    #                         252: [245, 150, 100],
    #                         256: [255, 0, 0],
    #                         253: [200, 40, 255],
    #                         254: [30, 30, 255],
    #                         255: [90, 30, 150],
    #                         257: [250, 80, 100],
    #                         258: [180, 30, 80],
    #                         259: [255, 0, 0]
    #     }
    # else:
    #     raise ValueError('Unsupport dataset:', dataset)
    # # plot
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # lims = int(len(label_to_colour)*1.6)
    # ax.axis([0, 4, 0, lims])
    # count = 0
    # for id, color in label_to_colour.items():
    #     y = lims - (count*3+1)
    #     if y > 0:
    #         x = 0.5
    #     else:
    #         x = 2.5
    #         y = -y
    #     ax.plot([x], [y], '.', color=(color[0]/255.0, color[1]/255.0, color[2]/255.0),  
    #             markersize=40) 
    #     ax.text(x+0.25, y-0.5, label_to_names[id], fontsize=15)
    #     count += 1
    # plt.show()


    #### plot loss ####
    # Log_folders = ['Recog_Log_2021-06-21_05-49-38', 'Recog_Log_2021-06-23_03-51-32', 'Recog_Log_2021-06-21_05-17-29', 'Recog_Log_2021-06-21_12-56-15']
    Log_folders = ['Recog_Log_2021-06-29_12-22-03']
    Log_legends = ['SGD, 5_feat, 6_neg', 'Adam, 5_feat, 6_neg', 'Adam, 3_feat, 8_neg', 'Adam, 5_feat, 6_neg, full epoch']
    Log_colors = ['r', 'g', 'b', 'y', 'c']
    avg_step = 100
    all_x = []
    all_y = []
    for i, folder in enumerate(Log_folders):
        steps = []
        loss = []
        count = 1
        with open('results/'+folder+'/training.txt') as f:
            lines = f.readlines()
            tmp = 0
            for line in lines:
                line = line.rstrip().split(' ')
                epoch = int(line[0][1:4])
                # one_loss = float(line[1][5:])
                # one_loss = float(line[1][2:-1])
                one_loss = float(line[1][2:])
                tmp += one_loss
                if count % avg_step ==0:
                    steps.append(count/avg_step)
                    loss.append(tmp/avg_step)
                    tmp = 0
                count += 1
        x = np.array(steps)
        y = np.array(loss)
        print(np.max(y))

        plt.plot(x, y, Log_colors[i], label=Log_legends[i])
    plt.legend()
    plt.title('Loss per epoch')
    plt.xlabel('Epoches')
    plt.xlim([0, 50])
    plt.ylabel('Triplet Loss')
    plt.show()


    # #### voxel grid downsample ####
    # pcd_name = 'scene0000_00_3900'
    # hd_scene_pcd = o3d.io.read_point_cloud('/home/yohann/Documents/test_ply/'+pcd_name+'.ply')
    # # voxel grid downsample the point cloud
    # cam_pts = np.asarray(hd_scene_pcd.points).astype(np.float32)
    # cam_rgb = np.asarray(hd_scene_pcd.colors)*255.
    # cam_rgb = cam_rgb.astype(np.uint8)
    # cam_label = np.zeros((cam_rgb.shape[0], ), dtype=np.int32)
    # sub_pts, sub_rgb, sub_lbls = grid_subsampling(cam_pts.astype(np.float32),
    #                                             features=cam_rgb.astype(np.float32),
    #                                             labels=cam_label.astype(np.int32),
    #                                             sampleDl=0.08)
    # # save as ply
    # write_ply('/home/yohann/Documents/test_ply/'+pcd_name+'_008.ply',
    #         (sub_pts.astype(np.float32), sub_rgb.astype(np.uint8), sub_lbls.astype(np.int32)), 
    #         ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])


    # #### File Manipulation ####
    # scannet_pcd_path = '/media/yohann/Datasets/datasets/ScanNet/scans/input_pcd'
    # for scene in os.listdir(scannet_pcd_path):
    #     # if scene == 'scene0000_00':
    #     if 'scene' in scene:
    #         print(scene)
    #         scene_pcd_path = os.path.join(scannet_pcd_path, scene)
    #         scene_ori_pcd_path = os.path.join(scene_pcd_path, 'pcd')
    #         # for (dirpath, dirname, filenames) in os.walk(scene_pcd_path):
    #         #     for filename in filenames:
    #         for pcd_file in os.listdir(scene_pcd_path):
    #             if '.ply' in pcd_file:
    #                 if 'sub' not in pcd_file:
    #                     if not os.path.exists(scene_ori_pcd_path):
    #                         os.mkdir(scene_ori_pcd_path)
    #                     shutil.move(os.path.join(scene_pcd_path, pcd_file), scene_ori_pcd_path)


    # #### Batch/Neighbor limit check ####
    # path_local = '/media/yohann/Datasets/datasets/ScanNet'
    # path_server = '/media/yohann/Datasets/datasets'
    # # load local file
    # batch_local_file = os.path.join(path_local, 'neighbors_limits.pkl')
    # if os.path.exists(batch_local_file):
    #     with open(batch_local_file, 'rb') as file:
    #         batch_local_dict = pickle.load(file)
    # else:
    #     batch_local_dict = {}
    # # load server file
    # batch_server_file = os.path.join(path_server, 'neighbors_limits.pkl')
    # if os.path.exists(batch_server_file):
    #     with open(batch_server_file, 'rb') as file:
    #         batch_server_dict = pickle.load(file)
    # else:
    #     batch_server_dict = {}
    
    # # update server file
    # for key, val in batch_local_dict.items():
    #     if key not in batch_server_dict.keys():
    #         batch_server_dict[key] = val
    # with open(batch_server_file, 'wb') as file:
    #     pickle.dump(batch_server_dict, file)

    # paths = ['/media/yohann/Datasets/datasets', '/media/yohann/Datasets/datasets/ScanNet']
    # for path in paths:
    #     print(path)
    #     # Load batch_limit dictionary
    #     batch_lim_file = os.path.join(path, 'batch_limits.pkl')
    #     if os.path.exists(batch_lim_file):
    #         with open(batch_lim_file, 'rb') as file:
    #             batch_lim_dict = pickle.load(file)
    #     else:
    #         batch_lim_dict = {}
    #     for key, val in batch_lim_dict.items():
    #         print(key, ':', val)

    #     # Load neighb_limits dictionary
    #     neighb_lim_file = os.path.join(path, 'neighbors_limits.pkl')
    #     if os.path.exists(neighb_lim_file):
    #         with open(neighb_lim_file, 'rb') as file:
    #             neighb_lim_dict = pickle.load(file)
    #     else:
    #         neighb_lim_dict = {}
    #     for key, val in neighb_lim_dict.items():
    #         print(key, ':', val)
    #     print('')


    # #### Point cloud io ####
    # # path = '/home/yohann/Documents/PontnetVLAD-data-test'
    # path = '/media/yohann/Datasets/datasets/ScanNet/scans/input_new_pcd/scene0000_00'
    # # ## one pcd a time
    # # folders = ['pointcloud_20m_10overlap', 'pointcloud_25m_10']
    # # for dir in os.listdir(path):
    # #     if not 'university' in dir:
    # #         continue
    # #     scene_path = os.path.join(path, dir)
    # #     print(scene_path)
    # #     for fdr in os.listdir(scene_path):
    # #         if fdr in folders:
    # #             for pcd in os.listdir(os.path.join(scene_path, fdr)):
    # #                 points = np.fromfile(os.path.join(os.path.join(scene_path, fdr, pcd)))
    # #                 points = points.reshape((points.shape[0]//3, 3))
    # #                 print(points.shape)
    # #                 o3dpcd = o3d.geometry.PointCloud()
    # #                 o3dpcd.points = o3d.utility.Vector3dVector(points)
    # #                 vis = o3d.visualization.Visualizer()
    # #                 vis.create_window(window_name='Submap', width=960, height=960, left=360, top=0)
    # #                 vis.add_geometry(o3dpcd)
    # #                 vis.run()
    # #                 vis.destroy_window()
    # ## 4 pcds
    # # path = os.path.join(path, '2014-05-19-13-20-57', 'pointcloud_20m_10overlap')
    # # file_names = ['1400505893170765.bin', '1400505894395159.bin', 
    # #               '1400505895618465.bin', '1400505896820388.bin']
    # file_names = ['scene0000_00_0_sub.ply', 'scene0000_00_390_sub.ply', 
    #               'scene0000_00_615_sub.ply', 'scene0000_00_720_sub.ply']
    
    # # points = np.fromfile(os.path.join(os.path.join(path, file_names[0])))
    # # points = points.reshape((points.shape[0]//3, 3))
    # # o3dpcd0 = o3d.geometry.PointCloud()
    # o3dpcd0 = o3d.io.read_point_cloud(os.path.join(path, file_names[0]))
    # # o3dpcd0.points = o3d.utility.Vector3dVector(points)
    # vis0 = o3d.visualization.Visualizer()
    # vis0.create_window(window_name=file_names[0], width=960, height=540, left=0, top=0)
    # vis0.add_geometry(o3dpcd0)

    # # points = np.fromfile(os.path.join(os.path.join(path, file_names[1])))
    # # points = points.reshape((points.shape[0]//3, 3))
    # # o3dpcd1 = o3d.geometry.PointCloud()
    # o3dpcd1 = o3d.io.read_point_cloud(os.path.join(path, file_names[1]))
    # # o3dpcd1.points = o3d.utility.Vector3dVector(points)
    # vis1 = o3d.visualization.Visualizer()
    # vis1.create_window(window_name=file_names[1], width=960, height=540, left=960, top=0)
    # vis1.add_geometry(o3dpcd1)

    # # points = np.fromfile(os.path.join(os.path.join(path, file_names[2])))
    # # points = points.reshape((points.shape[0]//3, 3))
    # # o3dpcd2 = o3d.geometry.PointCloud()
    # o3dpcd2 = o3d.io.read_point_cloud(os.path.join(path, file_names[2]))
    # # o3dpcd2.points = o3d.utility.Vector3dVector(points)
    # vis2 = o3d.visualization.Visualizer()
    # vis2.create_window(window_name=file_names[2], width=960, height=540, left=0, top=540)
    # vis2.add_geometry(o3dpcd2)

    # # points = np.fromfile(os.path.join(os.path.join(path, file_names[3])))
    # # points = points.reshape((points.shape[0]//3, 3))
    # # o3dpcd3 = o3d.geometry.PointCloud()
    # o3dpcd3 = o3d.io.read_point_cloud(os.path.join(path, file_names[3]))
    # # o3dpcd3.points = o3d.utility.Vector3dVector(points)
    # vis3 = o3d.visualization.Visualizer()
    # vis3.create_window(window_name=file_names[3], width=960, height=540, left=960, top=540)
    # vis3.add_geometry(o3dpcd3)

    
    # while True:
    #     vis0.update_geometry(o3dpcd0)
    #     if not vis0.poll_events():
    #         break
    #     vis0.update_renderer()

    #     vis1.update_geometry(o3dpcd1)
    #     if not vis1.poll_events():
    #         break
    #     vis1.update_renderer()

    #     vis2.update_geometry(o3dpcd2)
    #     if not vis2.poll_events():
    #         break
    #     vis2.update_renderer()

    #     vis3.update_geometry(o3dpcd3)
    #     if not vis3.poll_events():
    #         break
    #     vis3.update_renderer()
    # vis0.destroy_window()
    # vis1.destroy_window()
    # vis2.destroy_window()
    # vis3.destroy_window()

                


        


        
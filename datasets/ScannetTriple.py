#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling SemanticKitti dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Yuhang Ming
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import enum
import time
from matplotlib.pyplot import subplot_tool
import numpy as np
import pickle
import json

from numpy.lib.shape_base import _hvdsplit_dispatcher
import torch
import math
# import yaml
from multiprocessing import Lock


# OS functions
from os import listdir
from os.path import exists, join, isdir

# Dataset parent class
from datasets.common import PointCloudDataset
from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *

from utils.mesh import rasterize_mesh
from utils.metrics import fast_confusion

from datasets.common import grid_subsampling
from utils.config import bcolors

# Open3d for generating sub pcds
import open3d as o3d

# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class ScannetTripleDataset(PointCloudDataset):
    """Class to handle Scannet dataset for Triple segmentation."""

    def __init__(self, config, set='training', balance_classes=True):
        PointCloudDataset.__init__(self, 'ScannetTriple')

        ##########################
        # Parameters for the files
        ##########################

        # Dataset folder
        self.path = '/media/yohann/fastStorage/data/ScanNetPR'
        
        # # pcd without zero-meaning coordinates
        # self.input_pcd_path = join(self.path, 'scans', 'input_pcd')
        # pcd with zero-meaning coordinates
        self.data_path = join(self.path, 'scans')
        # self.input_pcd_path = join(self.path, 'scans', 'input_pcd_0mean')
        print("point cloud path:", self.data_path)

        # Type of task conducted on this dataset
        self.dataset_task = 'registration'

        # Training or test set
        self.set = set

        # Get a list of sequences
        # data_split_path = join(self.path, "test_files")
        data_split_path = join(self.path, "tools/Tasks/Benchmark")
        # data_split_path = join(self.path, "Tasks/Benchmark")
        # Cloud names
        if self.set == 'training':
            scene_file_name = join(data_split_path, 'scannetv2_train.txt')
            self.scenes = np.sort(np.loadtxt(scene_file_name, dtype=np.str))
        elif self.set == 'validation':
            scene_file_name = join(data_split_path, 'scannetv2_val.txt')
            self.scenes = np.sort(np.loadtxt(scene_file_name, dtype=np.str))
            # self.clouds = [self.scenes[0]]  # test
        elif self.set == 'test':
            scene_file_name = join(data_split_path, 'scannetv2_test.txt')
            # scene_file_name = join(data_split_path, 'scannetv2_test_val.txt')
            self.scenes = np.loadtxt(scene_file_name, dtype=np.str)
            self.scenes = self.scenes
            # self.scenes = [self.scenes[0]]  # only test one scene
            # print((self.scenes))
        else:
            raise ValueError('Unsupport set type')

        ###########################
        # Object classes parameters
        ###########################

        # Dict from labels to names
        # subset of 20 classes from NYUv2's 40 classes
        self.label_to_names = {0: 'unclassified',
                               1: 'wall',
                               2: 'floor',
                               3: 'cabinet',
                               4: 'bed',
                               5: 'chair',
                               6: 'sofa',
                               7: 'table',
                               8: 'door',
                               9: 'window',
                               10: 'bookshelf',
                               11: 'picture',
                               12: 'counter',
                               14: 'desk',
                               16: 'curtain',
                               24: 'refridgerator',
                               28: 'shower curtain',
                               33: 'toilet',
                               34: 'sink',
                               36: 'bathtub',
                               39: 'other furniture'
                               }
        # Dict from labels to colours
        # color values for each class TO BE ADDED
        self.label_to_colour = {0:  [0,   0,   0],   # black -> 'unclassified'
                                1:  [174, 198, 232], # light purple -> 'wall',
                                2:  [151, 223, 137], # lime -> 'floor',
                                3:  [31,  120, 180], # dark blue -> 'cabinet'
                                4:  [255, 188, 120], # light orange -> 'bed',
                                5:  [188, 189, 35],  #  -> 'chair',
                                6:  [140, 86,  74],  # brown -> 'sofa',
                                7:  [255, 152, 151], # pink -> 'table',
                                8:  [213, 39,  40],  # red -> 'door',
                                9:  [196, 176, 213], # light purple -> 'window',
                                10: [150, 102, 188], # purple -> 'bookshelf',
                                11: [196, 156, 148], # light brown -> 'picture',
                                12: [23,  190, 208], # dark cyan -> 'counter',
                                14: [247, 183, 210], # light pink -> 'desk',
                                16: [218, 219, 141], #  -> 'curtain',
                                24: [254, 127, 14],  # orange -> 'refridgerator',
                                28: [158, 218, 229], # light cyan -> 'shower curtain',
                                33: [43,  160, 45],  # green -> 'toilet',
                                34: [112, 128, 144], # grey -> 'sink',
                                36: [227, 119, 194], #  -> 'bathtub',
                                39: [82,  83,  163], # dark purple -> 'otherfurniture'
                               }

        # Initialize a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array([0])

        ##################
        # Other parameters
        ##################

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes - len(self.ignored_labels)  # ScanNet
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Potential like cloud segmentation?
        # # Using potential or random epoch generation
        # self.use_potentials = use_potentials

        #####################
        # Prepare point cloud
        #####################
        # self.intrinsics = []    # list of Ks for every scene
        # self.nframes = []       # list of nums of frames for every scene
        self.fids = []          # list of list of actual frame id used to create pcd
        self.poses = []         # list of list of frame pose used to create pcd
        self.files = []         # list of list of pcd files created, pts in camera coordinate frame
        # training/val only
        # self.pos_thred = 2**2
        self.posIds = []        # list of dictionary of positive pcd example ids for training
        self.negIds = []        # list of dictionary of negative pcd example ids for training
        self.pcd_sizes = []     # list of list of pcd sizes
        # val/testing only
        self.class_proportions = None
        self.val_confs = []     # for validation

        # Choose batch_num in_R and max_in_p depending on validation or training
        if self.set == 'training':
            self.batch_num = config.batch_num
            self.max_in_p = config.max_in_points
            self.in_R = config.in_radius
        else:
            # Loaded from training parameters
            self.batch_num = config.val_batch_num
            self.max_in_p = config.max_val_points
            self.in_R = config.val_radius

        # load sub cloud from the HD mesh w.r.t. cameras
        self.prepare_point_cloud()

        # get all_inds as 2D array
        # (index of the scene, index of the frame)
        seq_inds = np.hstack([np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.fids)])
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for _ in self.fids])
        # NOTE 2nd is the index of frames NOT the actual frame id
        self.all_inds = np.vstack((seq_inds, frame_inds)).T 

        ##### MODIFY HERE????
        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize frame potentials with random values
        self.potentials = torch.from_numpy(np.random.rand(self.all_inds.shape[0]) * 0.1 + 0.1)
        self.potentials.share_memory_()

        # If true, the same amount of frames is picked per class
        # SET FALSE HERE
        self.balance_classes = balance_classes

        # shared epoch indices and classes (in case we want class balanced sampler)
        if set == 'training':
            N = int(np.ceil(config.epoch_steps * self.batch_num * 1.1))
        else:
            N = int(np.ceil(config.validation_size * self.batch_num * 1.1))
        self.num_neg_samples = config.num_neg_samples
        # print(config.validation_size)
        # print(self.batch_num)
        # print('N = ', N)

        # current epoch id
        self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
        # index generated this epoch to get the desired point cloud
        # epoch should have length of at least epoch_steps * batch_num
        # with values from 0 - all_inds.shape[0] (initialised as 0s)
        self.epoch_inds = torch.from_numpy(np.zeros((N,), dtype=np.int64))
        # self.epoch_labels = torch.from_numpy(np.zeros((N,), dtype=np.int32))
        # print('\nPotential Info:')
        # print(self.potentials.size())  # size of the total pcd, #_scene * #_frame
        # print(self.epoch_i.size())     # counter, single value
        # print(self.epoch_inds.size())  # total selected center, >= batch_num * epoch_step

        self.epoch_i.share_memory_()
        self.epoch_inds.share_memory_()
        # self.epoch_labels.share_memory_()

        # multi-threading in data loading
        self.worker_waiting = torch.tensor([0 for _ in range(config.input_threads)], dtype=torch.int32)
        self.worker_waiting.share_memory_()
        self.worker_lock = Lock()

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return 0
        # return len(self.frames)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        # Initiate concatanation lists
        p_list = []     # points
        f_list = []     # features
        l_list = []     # labels

        fi_list = []    # current scene index and frame index
        p0_list = []    # center of current point cloud in CAM coordinate frame
        s_list = []     # scales
        R_list = []     # Rotations
        
        # scale and rot are set to identity here
        # check later
        r_inds_list = []
        r_mask_list = []

        val_labels_list = []    # gt label for val set
        batch_n = 0     # cumulative number of pts in current the batch

        while True:

            # Use potential minimum to get index of scene and frame
            with self.worker_lock:
                # Get potential minimum
                ind = int(self.epoch_inds[self.epoch_i])
                # print('chosen index:', ind)
                if ind < 0:
                    return []
                # wanted_label = int(self.epoch_labels[self.epoch_i])
                # Update epoch indice
                self.epoch_i += 1

            # Current/query pcd indices
            s_ind, f_ind = self.all_inds[ind]

            if self.set in ['training', 'validation']:
                ## items should be generated here:
                # reference to pointnet_vlad: train_pointnetvlad.py
                # current pcd index:        s_ind & f_ind;
                # positive pcd indices:     [pos_s_inds, pos_f_inds]; default 2
                # negative pcd indices:     [neg_s_inds, neg_f_inds]; default 6 (4 in dh3d)
                # other negative pcd index: o_s_ind, o_f_ind.

                # check there are enough positive pcds
                num_pos_ids = len(self.posIds[s_ind][f_ind])
                if num_pos_ids < 2:
                    print('Skip current pcd (', self.files[s_ind][f_ind].split('/')[-1], ') due to empty positives.')
                    return []
                # print('Current pcd index:', s_ind, f_ind)
                # print('Positive lists:', self.posIds[s_ind][f_ind])
                # print('Negative lists:', self.negIds[s_ind][f_ind])

                # Positive pcd indices
                # pos_s_ind = s_ind
                # tmp_f_inds = np.random.randint(0, num_pos_ids, 2)
                # pos_f_inds = np.array(self.posIds[s_ind][f_ind])[tmp_f_inds]
                pos_f_inds = [np.random.choice( self.posIds[s_ind][f_ind] )] 
                # 2 positives, ensure not choose the same positive pcd
                while True:
                    tmp = np.random.choice( self.posIds[s_ind][f_ind] ) 
                    if tmp != pos_f_inds[0]:
                        pos_f_inds.append(tmp)
                        break
                pos_f_inds = np.array(pos_f_inds)
                # print(num_pos_ids, pos_s_ind, pos_f_inds)
                all_indices = [ (s_ind, f_ind), (s_ind, pos_f_inds[0]), (s_ind, pos_f_inds[1]) ]
                
                ## Add selection criterion based on pcd size due GPU memory limitation ##
                # check pcd sizes, choose 2x(9000+), 1x(7000-9000), 1x(5000-7000), 2x(5000-)
                count_XL = 0
                max_XL = 1
                count_L = 0
                max_L = 1
                count_M = 0
                max_M = 1
                # choose at most one from the same scene
                if len(self.negIds[s_ind][f_ind]) > 0:
                    neg_s_inds = [s_ind]
                    neg_f_inds = [np.random.choice(self.negIds[s_ind][f_ind])]
                    # get the size of selected pcd
                    tmp_size = int(self.pcd_sizes[neg_s_inds[-1]][neg_f_inds[-1]])
                    # print(tmp_size)
                    if tmp_size >= 9000:
                        count_XL += 1
                    elif tmp_size >= 7000:
                        count_L += 1
                    elif tmp_size > 5000:
                        count_M += 1
                    else:
                        pass
                else:
                    neg_s_inds = []
                    neg_f_inds = []
                # continue select the rest negative pcds from other scenes
                while len(neg_s_inds) < self.num_neg_samples:
                    tmp_neg_s = np.random.choice( len(self.scenes) )
                    if self.scenes[s_ind][:9] != self.scenes[tmp_neg_s][:9]:
                        tmp_neg_f = np.random.randint(0, len(self.fids[tmp_neg_s]))
                        # check the pcd size
                        tmp_size = int(self.pcd_sizes[tmp_neg_s][tmp_neg_f])
                        if tmp_size >= 9000:
                            if count_XL == max_XL:
                                continue
                            count_XL += 1
                        elif tmp_size >= 7000:
                            if count_L == max_L:
                                continue
                            count_L += 1
                        elif tmp_size >= 5000:
                            if count_M == max_M:
                                continue
                            count_M += 1
                        else:
                            pass
                        # print(tmp_size)
                        neg_s_inds.append(tmp_neg_s)
                        neg_f_inds.append(tmp_neg_f)
                
                # print(neg_s_inds, neg_f_inds)
                for idx in range(self.num_neg_samples):
                    all_indices.append( (neg_s_inds[idx], neg_f_inds[idx]) )
                
                # Other negative pcd index for quadruplet
                # find a pcd that is negative to all anchor, positives, and negatives
                neg_star_s_ind = -1
                while neg_star_s_ind < 0:
                    tmp_neg = np.random.choice( len(self.scenes) )
                    bMatched = False
                    if self.scenes[s_ind][:9] != self.scenes[tmp_neg][:9]:
                        bMatched = True
                    for neg_s_ind in neg_s_inds:
                        if self.scenes[s_ind][:9] != self.scenes[tmp_neg][:9]:
                            bMatched = True
                            break
                    if not bMatched:
                        neg_star_s_ind = tmp_neg
                neg_star_f_ind = np.random.randint(0, len(self.fids[neg_star_s_ind]))
                all_indices.append( (neg_star_s_ind, neg_star_f_ind) )
                # print('All chosen indices: [query/current], [positive]*2, [negative]*4', all_indices)
            else:
                # in testing, only get the current index
                all_indices = [(s_ind, f_ind)]

            for s_ind, f_ind in all_indices:
                #################
                # Load the points
                # NOTE all points are in camera 
                #      coordinate frame
                #################

                # print(s_ind, f_ind, len(self.files), len(self.files[s_ind]))
                current_file = self.files[s_ind][f_ind]
                # print('Loading: ', current_file)

                o_pts = None
                o_labels = None

                # subpcd_file = current_file[:-4]+'_sub.ply'
                data = read_ply(current_file)
                sub_pts = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T # Nx3
                # sub_pts = sub_pts.astype(np.float32)
                if sub_pts.shape[0] < 2:
                    raise ValueError("Empty Polygan Mesh !!!!")
                # print(' -', current_file)
                sub_rgb = np.vstack((data['red'], data['green'], data['blue'])).astype(np.float32).T
                # sub_rgb = sub_rgb.astype(np.float32)
                sub_lbls = data['class'].astype(np.int32)  # zeros for test set
                # Get center of the first frame in camera coordinates
                p0 = np.mean(sub_pts, axis=0)
                # Convert p0 to world coordinates
                # in case of Matrix x Vector, np.dot = np.matmul
                crnt_pose = self.poses[s_ind][f_ind]
                p0 = crnt_pose[:3, :3] @ p0 + crnt_pose[:3, 3]

                # rescale float color and squeeze label
                sub_rgb = sub_rgb / 255.
                sub_lbls = np.squeeze(sub_lbls)     # eg. from shape (1, 3, 1) to shape (3,), axis to be squeezed out must have size 1

                # Number collected
                n = sub_pts.shape[0]

                # Randomly drop some points (augmentation process and safety for GPU memory consumption)
                # max_in_p is calibrated in training and load back in in testing
                # print('max_in_p:', self.max_in_p)
                if n > self.max_in_p:
                    input_inds = np.random.choice(n, size=self.max_in_p, replace=False)
                    sub_pts = sub_pts[input_inds, :]
                    sub_rgb = sub_rgb[input_inds, :]
                    sub_lbls = sub_lbls[input_inds]
                    n = input_inds.shape[0]

                #### ONLY used for segmentation test.
                #### !!!! DONT NEED THESE FOR VLAD TEST //- Yohann
                proj_inds = np.zeros((0,))
                reproj_mask = np.zeros((0,))

                ## sub points in camera coordinate system
                ## AUGMENTATION DISABLED FOR NOW
                # # Data augmentation
                # in_pts, scale, R = self.augmentation_transform(in_pts)
                scale = np.ones(3)
                R = np.eye(3)

                # Color augmentation
                if np.random.rand() > self.config.augment_color:
                    sub_rgb[:, 3:] *= 0

                # Stack batch
                p_list += [sub_pts]
                f_list += [sub_rgb]
                l_list += [sub_lbls]
                fi_list += [[s_ind, f_ind]]
                p0_list += [p0]
                s_list += [scale]
                R_list += [R]
                # don't need following three for VLAD
                r_inds_list += [proj_inds]
                r_mask_list += [reproj_mask]
                val_labels_list += [o_labels]

                # Update batch size
                batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        frame_inds = np.array(fi_list, dtype=np.int32)
        frame_centers = np.stack(p0_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # print("\nBatch Info: ")
        # print('points:', stacked_points.shape,  type(stacked_points[0,0]))
        # print('labels:', labels.shape,    type(labels[0]))
        # print('index:', frame_inds.shape, type(frame_inds[0]))
        # print('centers:', frame_centers.shape, type(frame_centers[0]))
        # print('lengths:', stack_lengths.shape, type(stack_lengths[0]))
        # print('scales:', scales.shape, type(scales[0]))
        # print('Rots:', rots.shape, type(rots[0]))

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        # print('colors:', features.shape,  type(features[0,0]))
        # print('unit:', stacked_features.shape,  type(stacked_features[0,0]))
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            # [1, r, g, b]
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        else: 
            raise ValueError('Only accepted input dimensions as 1 or 4 (without XYZ)')
        # print('features:', stacked_features.shape,  type(stacked_features[0,0]))


        #######################
        # Create network inputs
        #######################
        #
        #   points coordinates;
        #   neighbors, pooling, upsampling indices;
        #   length of input points;
        #   features, semantic labels;
        #   for each layers
        #
        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels.astype(np.int64),
                                              stack_lengths)

        # Add scale and rotation for testing
        input_list += [scales, rots, frame_inds, frame_centers, r_inds_list, r_mask_list, val_labels_list]

        # Display timings
        ## ADD LATER

        return [self.config.num_layers] + input_list

    def prepare_point_cloud(self):
        """
        generate sub point clouds from the complete
        reconstructed scene, using current pose and
        depth frame
        """

        if not exists(self.data_path):
            raise ValueError('Missing input pcd folder:', self.data_path)
        
        # Load pre-processed files [all tasks are stored in a single file]
        # positive and negative indices for each pcd
        vlad_pn_file = join(self.path, 'VLAD_triplets', 'vlad_pos_neg.pkl')
        with open(vlad_pn_file, "rb") as f:
            # dict, key = scene string, val = list of pairs of (list pos, list neg)
            all_scene_pos_neg = pickle.load(f)
        # zero-meaned pcd file names for each pcd
        valid_pcd_file = join(self.path, 'VLAD_triplets', 'vlad_pcd.pkl')
        with open(valid_pcd_file, "rb") as f:
            # dict, key = scene string, val = list of filenames
            all_scene_pcds = pickle.load(f)
        # num of pts in each pcd
        pcd_size_file = join(self.path, 'VLAD_triplets', 'pcd_size.pkl')
        with open(pcd_size_file, "rb") as f:
            # dict, key = scene string, val = list of point numbers
            dict_pcd_size = pickle.load(f)

        # Loop through all scenes to retrieve data for current task
        pcd_count = 0
        for i, scene in enumerate(self.scenes):
            pcd_count += len(all_scene_pcds[scene])
            print('{}%'.format(int(100*i/len(self.scenes))), flush=True, end='\r')
            # print('scene_id_name', i, scene, 'num_of_pcds', num_scene_pcds)
            # print('Processing:', scene, '(', i+1 , '/', len(self.scenes), ')') 
            # # print('  from', scene_folder)
            # # print('   to ', scene_pcd_path)
            
            # path to original ScanNet data
            scene_folder = join(self.data_path, scene)
            # path to processed ScanNet point cloud
            scene_pcd_path = join(scene_folder, 'input_pcd_0mean')
            if not exists(scene_pcd_path):
                raise ValueError('Missing scene folder:', scene_pcd_path)

            # get necessary data
            scene_files = []    # list of pcd file names for current scene
            scene_poses = []    # list of pcd poses for current scene
            scene_fids = []     # list of actual frame id used to generate the pcd
            all_posId = []      # list of positive pcd Indices
            all_negId = []      # list of negative pcd Indices
            for j, subpcd_file in enumerate(all_scene_pcds[scene]):
                actual_frame_id = int(subpcd_file[13:-8])
                # print('{}%'.format(int(100*j/num_scene_pcds)), flush=True, end='\r')
                frame_subpcd_file = join(scene_pcd_path, subpcd_file)

                # get pose of current frame
                pose = np.loadtxt(join(scene_folder, 'pose', str(actual_frame_id)+'.txt'))
                # double check if pose is lost
                chk_val = np.sum(pose)
                if np.isinf(chk_val) or np.isnan(chk_val):
                    raise ValueError('Invalid pose value for', scene_folder, actual_frame_id)

                # store file name info
                scene_files.append(frame_subpcd_file)
                scene_poses.append(pose)
                scene_fids.append(actual_frame_id)

                # store current pos and neg index
                # all_posId[j] = all_scene_pos_neg[scene][j][0]
                # all_negId[j] = all_scene_pos_neg[scene][j][1]
                all_posId.append(all_scene_pos_neg[scene][j][0])
                all_negId.append(all_scene_pos_neg[scene][j][1])

                # double check if the point cloud file exists
                if not exists(frame_subpcd_file):
                    raise ValueError('Missing subpcd file:', frame_subpcd_file)
            # print('100 %')

            self.files.append(scene_files)
            self.poses.append(scene_poses)
            self.fids.append(scene_fids)
            self.pcd_sizes.append(dict_pcd_size[scene])
            if self.set in ['training', 'validation']:
                self.posIds.append(all_posId)
                self.negIds.append(all_negId)
            # print(self.posIds)
        print('Total # of pcd:', pcd_count)

        if self.set in ['training', 'validation']:
            self.class_proportions = np.ones((self.num_classes,), dtype=np.int32)

    def parse_scene_info(self, filename):
        """ read information file with given filename

            Returns
            -------
            int 
                number of frames in the sequence
            list
                [height width fx fy cx cy].
        """
        K = []
        info_file = open(filename)
        for line in info_file:
            vals = line.strip().split(' ')

            if 'depth' in vals[0]:
                K.append(float(vals[2]))
            if vals[0] == 'numDepthFrames':
                nFrames = int(vals[2])
        return nFrames, K

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class ScannetTripleSampler(Sampler):
    """
    Sampler for ScannetTriple
    Generate the index for loading at each iteration/step
    """

    def __init__(self, dataset: ScannetTripleDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == 'training':
            self.N = dataset.config.epoch_steps
        else:
            self.N = dataset.config.validation_size

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """

        if self.dataset.balance_classes:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0
            # self.dataset.epoch_labels *= 0

            # Number of sphere centers taken per class in each cloud
            num_centers = self.dataset.epoch_inds.shape[0]

            # Generate a list of indices balancing classes and respecting potentials
            gen_indices = []
            gen_classes = []
            for i, c in enumerate(self.dataset.label_values):
                if c not in self.dataset.ignored_labels:

                    # Get the potentials of the frames containing this class
                    class_potentials = self.dataset.potentials[self.dataset.class_frames[i]]

                    # Get the indices to generate thanks to potentials
                    used_classes = self.dataset.num_classes - len(self.dataset.ignored_labels)
                    class_n = num_centers // used_classes + 1
                    if class_n < class_potentials.shape[0]:
                        _, class_indices = torch.topk(class_potentials, class_n, largest=False)
                    else:
                        class_indices = torch.zeros((0,), dtype=torch.int32)
                        while class_indices.shape < class_n:
                            new_class_inds = torch.randperm(class_potentials.shape[0])
                            class_indices = torch.cat((class_indices, new_class_inds), dim=0)
                        class_indices = class_indices[:class_n]
                    class_indices = self.dataset.class_frames[i][class_indices]

                    # Add the indices to the generated ones
                    gen_indices.append(class_indices)
                    gen_classes.append(class_indices * 0 + c)

                    # Update potentials
                    update_inds = torch.unique(class_indices)
                    self.dataset.potentials[update_inds] = torch.ceil(self.dataset.potentials[update_inds])
                    self.dataset.potentials[update_inds] += torch.from_numpy(np.random.rand(update_inds.shape[0]) * 0.1 + 0.1)

            # Stack the chosen indices of all classes
            gen_indices = torch.cat(gen_indices, dim=0)
            gen_classes = torch.cat(gen_classes, dim=0)

            # Shuffle generated indices
            rand_order = torch.randperm(gen_indices.shape[0])[:num_centers]
            gen_indices = gen_indices[rand_order]
            gen_classes = gen_classes[rand_order]

            # Update potentials (Change the order for the next epoch)
            #self.dataset.potentials[gen_indices] = torch.ceil(self.dataset.potentials[gen_indices])
            #self.dataset.potentials[gen_indices] += torch.from_numpy(np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1)

            # Update epoch inds
            self.dataset.epoch_inds += gen_indices
            # self.dataset.epoch_labels += gen_classes.type(torch.int32)

        else:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0
            # self.dataset.epoch_labels *= 0

            # Number of sphere centers taken in total
            num_centers = self.dataset.epoch_inds.shape[0]
            # print(num_centers, self.dataset.potentials.shape[0])

            # Get the list of indices to generate thanks to potentials
            if num_centers < self.dataset.potentials.shape[0]:
                # means more data than the number of centers used
                # pick top num_centers clouds
                # gen_indices has the length of "num_centers"
                _, gen_indices = torch.topk(self.dataset.potentials, num_centers, largest=False, sorted=True)
            else:
                # means the whole dataset is finished without the necessary steps
                if self.dataset.set in ['training', 'validation']:
                    gen_indices = torch.randperm(self.dataset.potentials.shape[0])
                else:
                    gen_indices = torch.arange(0, self.dataset.potentials.shape[0])

            # Update potentials (Change the order for the next epoch)
            self.dataset.potentials[gen_indices] = torch.ceil(self.dataset.potentials[gen_indices])
            self.dataset.potentials[gen_indices] += torch.from_numpy(np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1)

            # append with -1, stop this epoch once -1 is used to fecth new batch
            if num_centers >= self.dataset.potentials.shape[0]:
                app_indices = torch.from_numpy(-1 * np.ones(num_centers-self.dataset.potentials.shape[0]).astype(np.int64) ) 
                gen_indices = torch.cat((gen_indices, app_indices))
            # print(self.dataset.potentials.shape[0])
            print('num_centers =', num_centers)
            # print(gen_indices)
            # print(gen_indices.shape)
            self.dataset.epoch_inds += gen_indices
            print('epoch_inds:', self.dataset.epoch_inds)

        # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N

    def calib_max_in(self, config, dataloader, untouched_ratio=0.8, verbose=True, force_redo=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration of max_in_points value (use verbose=True for more details)')
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load max_in_limit dictionary
        max_in_lim_file = join(self.dataset.path, 'max_in_limits.pkl')
        if exists(max_in_lim_file):
            with open(max_in_lim_file, 'rb') as file:
                max_in_lim_dict = pickle.load(file)
        else:
            max_in_lim_dict = {}

        # Check if the max_in limit associated with current parameters exists
        if self.dataset.balance_classes:
            sampler_method = 'balanced'
        else:
            sampler_method = 'random'
        key = '{:s}_{:.3f}_{:.3f}'.format(sampler_method,
                                          self.dataset.in_R,
                                          self.dataset.config.first_subsampling_dl)
        if not redo and key in max_in_lim_dict:
            self.dataset.max_in_p = max_in_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check max_in limit dictionary')
            if key in max_in_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(max_in_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ########################
            # Batch calib parameters
            ########################

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            all_lengths = []
            N = 1000

            #####################
            # Perform calibration
            #####################

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Control max_in_points value
                    all_lengths += batch.lengths[0].tolist()

                    # Convergence
                    if len(all_lengths) > N:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if t - last_display > 1.0:
                        last_display = t
                        message = 'Collecting {:d} in_points: {:5.1f}%'
                        print(message.format(N,
                                             100 * len(all_lengths) / N))

                if breaking:
                    break

            self.dataset.max_in_p = int(np.percentile(all_lengths, 100*untouched_ratio))

            if verbose:

                # Create histogram
                a = 1

            # Save max_in_limit dictionary
            print('New max_in_p = ', self.dataset.max_in_p)
            max_in_lim_dict[key] = self.dataset.max_in_p
            with open(max_in_lim_file, 'wb') as file:
                pickle.dump(max_in_lim_dict, file)

        # Update value in config
        if self.dataset.set == 'training':
            config.max_in_points = self.dataset.max_in_p
        else:
            config.max_val_points = self.dataset.max_in_p

        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        if self.dataset.balance_classes:
            sampler_method = 'balanced'
        else:
            sampler_method = 'random'
        key = '{:s}_{:.3f}_{:.3f}_{:d}_{:d}'.format(sampler_method,
                                                    self.dataset.in_R,
                                                    self.dataset.config.first_subsampling_dl,
                                                    self.dataset.batch_num,
                                                    self.dataset.max_in_p)
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('Previous calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:s}_{:d}_{:.3f}_{:.3f}'.format(sampler_method, self.dataset.max_in_p, dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if not redo and len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:s}_{:d}_{:.3f}_{:.3f}'.format(sampler_method, self.dataset.max_in_p, dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.deform_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.batch_num

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Save input pointcloud sizes to control max_in_points
            cropped_n = 0
            all_n = 0

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            #####################
            # Perform calibration
            #####################

            #self.dataset.batch_limit[0] = self.dataset.max_in_p * (self.dataset.batch_num - 1)

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Control max_in_points value
                    are_cropped = batch.lengths[0] > self.dataset.max_in_p - 1
                    cropped_n += torch.sum(are_cropped.type(torch.int32)).item()
                    all_n += int(batch.lengths[0].shape[0])

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.frame_inds)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller (Increment)
                    self.dataset.batch_limit[0] += Kp * error

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                             estim_b,
                                             int(self.dataset.batch_limit[0])))

                if breaking:
                    break

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                         neighb_hists[layer, neighb_size],
                                                         bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Control max_in_points value
            print('\n**************************************************\n')
            if cropped_n > 0.3 * all_n:
                color = bcolors.FAIL
            else:
                color = bcolors.OKGREEN
            print('Current value of max_in_points {:d}'.format(self.dataset.max_in_p))
            print('  > {:}{:.1f}% inputs are cropped{:}'.format(color, 100 * cropped_n / all_n, bcolors.ENDC))
            if cropped_n > 0.3 * all_n:
                print('\nTry a higher max_in_points value\n'.format(100 * cropped_n / all_n))
                #raise ValueError('Value of max_in_points too low')
            print('\n**************************************************\n')

            # Save batch_limit dictionary
            key = '{:s}_{:.3f}_{:.3f}_{:d}_{:d}'.format(sampler_method,
                                                        self.dataset.in_R,
                                                        self.dataset.config.first_subsampling_dl,
                                                        self.dataset.batch_num,
                                                        self.dataset.max_in_p)
            batch_lim_dict[key] = float(self.dataset.batch_limit[0])
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:s}_{:d}_{:.3f}_{:.3f}'.format(sampler_method, self.dataset.max_in_p, dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)


        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class ScannetTripleCustomBatch:
    """
    Custom batch definition with memory pinning for ScannetTriple
    Originally a custom batch only has information of 1 point cloud
    """

    def __init__(self, input_list):
        # NOTE: points in camera coordinates
        #       centers in world coordinates

        # print(len(input_list[0]))
        # return empty batch to inform no more point left
        if len(input_list[0]) == 0:
            self.points = []
            self.neighbors = []
            self.pools = []
            self.upsamples = []
            self.lengths = []
            self.features = torch.empty(0)
            self.labels = torch.empty(0)
            self.scales = torch.empty(0)
            self.rots = torch.empty(0)
            self.frame_inds = torch.empty(0)
            self.frame_centers = torch.empty(0)
            self.reproj_inds = []
            self.reproj_masks = []
            self.val_labels = []
            return

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = int(input_list[0])

        # Extract input tensors from the list of numpy array
        ind = 1
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.frame_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.frame_centers = torch.from_numpy(input_list[ind])
        ind += 1
        self.reproj_inds = input_list[ind]
        ind += 1
        self.reproj_masks = input_list[ind]
        ind += 1
        self.val_labels = input_list[ind]

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.frame_inds = self.frame_inds.pin_memory()
        self.frame_centers = self.frame_centers.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.frame_inds = self.frame_inds.to(device)
        self.frame_centers = self.frame_centers.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def ScannetTripleCollate(batch_data):
    return ScannetTripleCustomBatch(batch_data)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Debug functions
#       \*********************/


def debug_timing(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.batch_num
    estim_N = 0

    for epoch in range(10):

        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.frame_inds) - estim_b) / 100
            estim_N += (batch.features.shape[0] - estim_N) / 10

            # Pause simulating computations
            time.sleep(0.05)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f} - {:.0f}'
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1],
                                     estim_b,
                                     estim_N))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_class_w(dataset, loader):
    """Timing of generator function"""

    i = 0

    counts = np.zeros((dataset.num_classes,), dtype=np.int64)

    s = '{:^6}|'.format('step')
    for c in dataset.label_names:
        s += '{:^6}'.format(c[:4])
    print(s)
    print(6*'-' + '|' + 6*dataset.num_classes*'-')

    for epoch in range(10):
        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # count labels
            new_counts = np.bincount(batch.labels)

            counts[:new_counts.shape[0]] += new_counts.astype(np.int64)

            # Update proportions
            proportions = 1000 * counts / np.sum(counts)

            s = '{:^6d}|'.format(i)
            for pp in proportions:
                s += '{:^6.1f}'.format(pp)
            print(s)
            i += 1


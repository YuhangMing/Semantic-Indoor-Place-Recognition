#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling Scannet dataset.
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
import time
import numpy as np
import pickle
import json
import torch
import math
from multiprocessing import Lock


# OS functions
from os import listdir
from os.path import exists, join, isfile, isdir

# Dataset parent class
from datasets.common import PointCloudDataset
from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *
from utils.mesh import rasterize_mesh

from datasets.common import grid_subsampling
from utils.config import bcolors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/

# Map-Style dataset, has __getitem__ & __len__
# eg. when accessed with dataset[idx], could read 
# the idx-th image and its corresponding label 
# from a folder on the disk.
class ScannetDataset(PointCloudDataset):
    """Class to handle Scannet dataset."""

    def __init__(self, config, set='training', use_potentials=True, load_data=True, load_test=False):
        """
        This dataset is LARGE, change from load all points to load as needed
        This dataset is small enough to be stored in-memory, 
        so load all point clouds here
        """
        PointCloudDataset.__init__(self, 'Scannet')

        ############
        # Parameters
        ############

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
        self.colour_to_label = {0:  [0,   0,   0],   # black -> 'unclassified'
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

        # Dataset folder
        # # self.path = '/home/yohann/NNs/data/ScanNet'
        self.path = '/media/yohann/My Passport/datasets/ScanNet'
        # # self.path = '/mnt/nas_7/datasets/ScanNet'
        # # test on seven scenes
        # self.path = '/media/yohann/My Passport/datasets/7Scenes/point_cloud'

        # Type of task conducted on this dataset
        self.dataset_task = 'cloud_segmentation'

        # Consider to change the task here
        # self.dataset_task = 'slam_segmentation'

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes - len(self.ignored_labels)
        config.dataset_task = self.dataset_task

        # (training) Parameters from config
        self.config = config

        # Training or test set
        self.set = set

        # Using potential or random epoch generation
        self.use_potentials = use_potentials

        # Number of models used per epoch
        # epoch_steps is the number of steps/forward_passes per epoch
        # batch_num is the number of (center) points processed per batch
        if self.set == 'training':
            self.epoch_n = config.epoch_steps * config.batch_num
        elif self.set in ['validation', 'test', 'ERF', 'visualise']:
            self.epoch_n = config.validation_size * config.batch_num
        else:
            raise ValueError('Unknown set for Scannet data: ', self.set)

        # Stop data is not needed
        if not load_data:
            return
        

        ###################
        # Prepare ply files
        ###################
        # Folder for the ply files
        self.ply_path = join(self.path, 'scans')
        # Path to the training/validation/test files
        self.finer_pc_path = join(self.ply_path, self.set+'_points')
        self.mesh_path = join(self.ply_path, self.set+'_meshes')

        data_split_path = join(self.path, "test_files")
        # data_split_path = join(self.path, "Tasks/Benchmark")
        # # Cloud names
        if self.set == 'training':
            scene_file_name = join(data_split_path, 'scannetv2_train.txt')
            self.clouds = np.sort(np.loadtxt(scene_file_name, dtype=np.str))
        elif self.set == 'validation':
            scene_file_name = join(data_split_path, 'scannetv2_val.txt')
            self.clouds = np.sort(np.loadtxt(scene_file_name, dtype=np.str))
            # self.clouds = [self.clouds[0]]  # test
        elif self.set == 'test':
            scene_file_name = join(data_split_path, 'scannetv2_test.txt')
            self.clouds = np.loadtxt(scene_file_name, dtype=np.str)
            # print((self.clouds))
            self.clouds = [self.clouds[0]]
            print((self.clouds))
            # self.clouds = [np.loadtxt(scene_file_name, dtype=np.str)]   # Single test file
        else:
            raise ValueError('Unsupport set type')

        # print(scene_file_name)
        # print(np.loadtxt(scene_file_name, dtype=np.str))
        # self.clouds = np.sort(np.loadtxt(scene_file_name, dtype=np.str))
        # for trainer only
        self.all_splits = [0, 1]
        self.validation_split = 1

        # prepare plygon mesh for all train/validation/test files
        ####### SCANNET #######
        self.prepare_Scannet_ply()
        ####### 7SCENES #######
        # self.prepare_room_ply('point_cloud_620')

        # list of training/validation/test files
        # self.files = np.sort([join(self.finer_pc_path, f) for f in listdir(self.finer_pc_path) if f[-4:] == '.ply'])
        self.files = [join(self.finer_pc_path, f+'.ply') for f in self.clouds]
        print(self.set, 'files: ', len(self.files))
        print(self.files)

        #### Parameters used in TF code, check how to use here
        # # 1 to do validation, 2 to train on all data
        # self.validation_split = 1
        # self.all_splits = []
        ######################################################

        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        #### Don't store all the data in these containers 
        #### If the dataset is too large to be loaded at once
        # Initiate containers
        self.input_trees = []
        self.input_colors = []
        self.input_vert_inds = []
        self.input_labels = []
        self.pot_trees = []
        self.num_clouds = 0
        self.validation_labels = []
        self.test_proj = []

        # Start loading
        # perform grid-based subsampling, create KDTree for it and
        # for each point in the original point cloud, find it a closet subsample point
        self.load_subsampled_clouds()
        # print('input trees: ', len(self.input_trees))
        # print('input colors: ', len(self.input_colors))
        # print('input vert inds: ', len(self.input_vert_inds))
        # print('input labels: ', len(self.input_labels))
        # print('pot trees: ', len(self.pot_trees))
        # print('val labels: ', len(self.validation_labels))
        # print('test proj_inds: ', len(self.test_proj))

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize potentials (with random numbers)
        if use_potentials:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []
            # for i, tree in enumerate(self.input_trees):   # tf2 version Scannet uses input tree
            for i, tree in enumerate(self.pot_trees):   # pytorch version S3DIS uses potential tree
                # np.random.rand() random number from [0, 1) with unform distribution
                # np.random.randn() random number from normal distribution N(0, 1)
                # so here, generated an 1D array of size point_num of
                # random_number/1000 and convert to torch tensor
                self.potentials += [torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)]
                
                # get minimum potentials and its index
                min_ind = int(torch.argmin(self.potentials[-1]))
                self.argmin_potentials += [min_ind]
                self.min_potentials += [float(self.potentials[-1][min_ind])]

            # Share potential memory
            self.argmin_potentials = torch.from_numpy(np.array(self.argmin_potentials, dtype=np.int64))
            self.min_potentials = torch.from_numpy(np.array(self.min_potentials, dtype=np.float64))
            self.argmin_potentials.share_memory_()
            self.min_potentials.share_memory_()
            # for i, _ in enumerate(self.input_trees):
            for i, _ in enumerate(self.pot_trees):
                self.potentials[i].share_memory_()

            self.worker_waiting = torch.tensor([0 for _ in range(config.input_threads)], dtype=torch.int32)
            self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0

        else:
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            N = config.epoch_steps * config.batch_num
            self.epoch_inds = torch.from_numpy(np.zeros((2, N), dtype=np.int64))
            self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()
        
        print('')
        print('potentials:', type(self.potentials), 'of', len(self.potentials), type(self.potentials[0]))
        print('idx:', len(self.argmin_potentials), self.argmin_potentials)
        print('min:', len(self.min_potentials), self.min_potentials)

        self.worker_lock = Lock()

        # For ERF visualization, we want only one cloud per batch and no randomness
        if self.set == 'ERF':
            self.batch_limit = torch.tensor([1], dtype=torch.float32)
            self.batch_limit.share_memory_()
            np.random.seed(42)

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.clouds)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        if self.use_potentials:
            return self.potential_item(batch_i)
        else:
            raise ValueError('Only support potential batch for now')
            # return self.random_item(batch_i)

    def potential_item(self, batch_i, debug_workers=False):
        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0

        info = get_worker_info()
        if info is not None:
            wid = info.id
        else:
            wid = None
        # print('\nwid = ', wid)

        while True:

            t += [time.time()]

            if debug_workers:
                message = ''
                for wi in range(info.num_workers):
                    if wi == wid:
                        message += ' {:}X{:} '.format(bcolors.FAIL, bcolors.ENDC)
                    elif self.worker_waiting[wi] == 0:
                        message += '   '
                    elif self.worker_waiting[wi] == 1:
                        message += ' | '
                    elif self.worker_waiting[wi] == 2:
                        message += ' o '
                print(message)
                self.worker_waiting[wid] = 0

            with self.worker_lock:

                if debug_workers:
                    message = ''
                    for wi in range(info.num_workers):
                        if wi == wid:
                            message += ' {:}v{:} '.format(bcolors.OKGREEN, bcolors.ENDC)
                        elif self.worker_waiting[wi] == 0:
                            message += '   '
                        elif self.worker_waiting[wi] == 1:
                            message += ' | '
                        elif self.worker_waiting[wi] == 2:
                            message += ' o '
                    print(message)
                    self.worker_waiting[wid] = 1

                # Get potential minimum
                cloud_ind = int(torch.argmin(self.min_potentials))
                point_ind = int(self.argmin_potentials[cloud_ind])

                # Get potential points from tree structure
                # return coarser subsampled data points
                pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)

                # Center point of input region
                center_point = pot_points[point_ind, :].reshape(1, -1)

                # Add a small noise to center point
                if self.set != 'ERF':
                    center_point += np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape)

                # Indices of points in input region
                # search for points in the spherical region given radius
                pot_inds, dists = self.pot_trees[cloud_ind].query_radius(center_point,
                                                                         r=self.config.in_radius,
                                                                         return_distance=True)

                d2s = np.square(dists[0])
                pot_inds = pot_inds[0]

                # Update potentials (Tukey weights)
                if self.set != 'ERF':
                    tukeys = np.square(1 - d2s / np.square(self.config.in_radius))
                    tukeys[d2s > np.square(self.config.in_radius)] = 0
                    self.potentials[cloud_ind][pot_inds] += tukeys
                    min_ind = torch.argmin(self.potentials[cloud_ind])
                    self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
                    self.argmin_potentials[[cloud_ind]] = min_ind

            t += [time.time()]

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)


            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            t += [time.time()]

            # Number collected
            n = input_inds.shape[0]

            # Collect labels and colors
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.set in ['test', 'ERF', 'visualise']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])

            t += [time.time()]

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0

            input_features = np.hstack((input_colors, input_points + center_point)).astype(np.float32)

            t += [time.time()]

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

            # Randomly drop some points (act as an augmentation process and a safety for GPU memory consumption)
            # if n > int(self.batch_limit):
            #    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
            #    n = input_inds.shape[0]

        # # test
        # print('batch: ', batch_n, self.batch_limit)
        
        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 5:
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')
        # in_features_dim == 4;
        # one_feat = [1, r, g, b]
        # print(stacked_features)

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        t += [time.time()]

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)

        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        if debug_workers:
            message = ''
            for wi in range(info.num_workers):
                if wi == wid:
                    message += ' {:}0{:} '.format(bcolors.OKBLUE, bcolors.ENDC)
                elif self.worker_waiting[wi] == 0:
                    message += '   '
                elif self.worker_waiting[wi] == 1:
                    message += ' | '
                elif self.worker_waiting[wi] == 2:
                    message += ' o '
            print(message)
            self.worker_waiting[wid] = 2

        t += [time.time()]

        # Display timings
        debugT = False
        if debugT:
            print('\n************************\n')
            print('Timings:')
            ti = 0
            N = 5
            mess = 'Init ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Pots ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Sphere .... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Collect ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Augment ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += N * (len(stack_lengths) - 1) + 1
            print('concat .... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('input ..... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('stack ..... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('\n************************\n')

        return input_list

    def random_item(self, batch_i):

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0

        while True:

            with self.worker_lock:

                # Get potential minimum
                cloud_ind = int(self.epoch_inds[0, self.epoch_i])
                point_ind = int(self.epoch_inds[1, self.epoch_i])

                # Update epoch indice
                self.epoch_i += 1

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

            # Add a small noise to center point
            if self.set != 'ERF':
                center_point += np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape)

            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            # Number collected
            n = input_inds.shape[0]

            # Collect labels and colors
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.set in ['test', 'ERF']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0

            # Get original height as additional feature
            input_features = np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

            # Randomly drop some points (act as an augmentation process and a safety for GPU memory consumption)
            # if n > int(self.batch_limit):
            #    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
            #    n = input_inds.shape[0]

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 5:
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        return input_list

    def prepare_Scannet_ply(self):

        print('\nPreparing ply files')
        t0 = time.time()

        # Mapping from annot to NYU labels ID
        label_files = join(self.ply_path, 'scannetv2-labels.combined.tsv')
        with open(label_files, 'r') as f:
            lines = f.readlines()
            names1 = [line.split('\t')[1] for line in lines[1:]]
            IDs = [int(line.split('\t')[4]) for line in lines[1:]]
            annot_to_nyuID = {n: id for n, id in zip(names1, IDs)}
        # print(annot_to_nyuID)   # a dictionary from object name to NYUv2 label ID
        # print(names1)
        # print(IDs)
        
        # for subset, new_path, mesh_path in zip(self.clouds, new_paths, mesh_paths):
        # Create folder
        if not exists(self.finer_pc_path):
            makedirs(self.finer_pc_path)
        if not exists(self.mesh_path):
            makedirs(self.mesh_path)

        N = len(self.clouds)
        
        for i, scene in enumerate(self.clouds):
            print('  processing: ', scene)
            ##############
            # Load meshes 
            ##############

            # Check if file already done
            if exists(join(self.finer_pc_path, scene + '.ply')):
                # vertex_data, faces = read_ply(join(self.mesh_path, scene + '_mesh.ply'), triangular_mesh=True)
                # for one_label in vertex_data['class']:
                #     print(one_label)
                # print(vertex_data['class'])
                # break
                continue
            t1 = time.time()

            # Read mesh
            # Add additional class label in the mesh file for training
            vertex_data, faces = read_ply(join(self.ply_path, scene, scene + '_vh_clean_2.ply'), triangular_mesh=True)
            vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
            # # test with vertices rotated 90 degrees around y axis
            # # corrupted output
            # print(vertices.shape)
            # print(vertices)
            # Ry90 = np.array([
            #     [0, 0, 1], [0, 1, 0], [-1, 0, 0]
            # ])
            # print(Ry90)
            # vertices = np.matmul(Ry90, vertices.T).T
            # print(vertices.shape)
            # print(vertices)
            vertices_colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T

            vertices_labels = np.zeros(vertices.shape[0], dtype=np.int32)
            if self.set == 'test':
                # Save mesh
                write_ply(join(self.mesh_path, scene + '_mesh.ply'),
                            [vertices, vertices_colors],
                            ['x', 'y', 'z', 'red', 'green', 'blue'],
                            triangular_faces=faces)
            else:
                # Load alignment matrix to realign points
                align_mat = None
                with open(join(self.ply_path, scene, scene + '.txt'), 'r') as txtfile:
                    lines = txtfile.readlines()
                for line in lines:
                    line = line.split()
                    if line[0] == 'axisAlignment':
                        align_mat = np.array([float(x) for x in line[2:]]).reshape([4, 4]).astype(np.float32)
                R = align_mat[:3, :3]
                T = align_mat[:3, 3]
                print(R, T)
                vertices = vertices.dot(R.T) + T

                # Get objects segmentations
                # over-segmentation of the annotation mesh
                # scene_vh_clean_2.0.010000.segs.json is for the low resolution mesh
                # scene_vh_clean.segs.json            is for the high resolution mesh
                with open(join(self.ply_path, scene, scene + '_vh_clean_2.0.010000.segs.json'), 'r') as f:
                    segmentations = json.load(f)
                print('  Object Segmentation: ', type(segmentations))
                print(segmentations.keys()) # a dictionary of 'params', 'sceneId', 'segIndices'

                segIndices = np.array(segmentations['segIndices'])
                # print(segIndices)       # assign a segment id to each point/vertex in the low res mesh
                # print(segIndices.shape) # same size as the number of points/vertices in low resolution mesh

                # Get objects classes
                # Aggregated instance-level semantic annotations on meshes
                # scene.aggregation.json            is for the low resolution mesh
                # scene_vh_clean.aggregation.json   is for the high resolution mesh
                with open(join(self.ply_path, scene, scene + '.aggregation.json'), 'r') as f:
                    aggregation = json.load(f)
                print('  Object Classes: ', type(aggregation))
                print(aggregation.keys())   # a dictionary of 'sceneId', 'appId', 'segGroups', 'segmentsFile'.
                                            # segGroup is a list of dictionaries, each dict contains
                                            # 'id', 'objId', 'segments', 'label', where 'segments' is a list 
                                            # of indices of the over-segments, and 'label' is the object label

                # Loop on object to classify points
                for segGroup in aggregation['segGroups']:
                    print(segGroup)
                    c_name = segGroup['label']
                    if c_name in names1:
                        nyuID = annot_to_nyuID[c_name]
                        if nyuID in self.label_values:
                            for segment in segGroup['segments']:
                                vertices_labels[segIndices == segment] = nyuID

                # Save mesh
                # write_ply(join(mesh_path, scene + '_mesh.ply'),
                #           [vertices, vertices_colors, vertices_labels],
                #           ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
                print(vertices_labels)
                write_ply(join(self.mesh_path, scene + '_mesh.ply'),
                            [vertices, vertices_colors, vertices_labels],
                            ['x', 'y', 'z', 'red', 'green', 'blue', 'class'],
                            triangular_faces=faces)
            
            # ###### TEST #######
            # # Redo for the high res mesh
            # # Read mesh
            # # Add additional class label in the mesh file for training
            # vertex_data, faces = read_ply(join(self.ply_path, scene, scene + '_vh_clean.ply'), triangular_mesh=True)
            # vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
            # vertices_colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T
            # vertices_labels = np.zeros(vertices.shape[0], dtype=np.int32)
            # if self.set == 'test':
            #     raise ValueError('yet to be implement')
            # else:
            #     # Load alignment matrix to realign points
            #     align_mat = None
            #     with open(join(self.ply_path, scene, scene + '.txt'), 'r') as txtfile:
            #         lines = txtfile.readlines()
            #     for line in lines:
            #         line = line.split()
            #         if line[0] == 'axisAlignment':
            #             align_mat = np.array([float(x) for x in line[2:]]).reshape([4, 4]).astype(np.float32)
            #     R = align_mat[:3, :3]
            #     T = align_mat[:3, 3]
            #     print(R, T)
            #     vertices = vertices.dot(R.T) + T
            #     # Get objects segmentations
            #     # over-segmentation of the annotation mesh
            #     # scene_vh_clean_2.0.010000.segs.json is for the low resolution mesh
            #     # scene_vh_clean.segs.json            is for the high resolution mesh
            #     with open(join(self.ply_path, scene, scene + '_vh_clean.segs.json'), 'r') as f:
            #         segmentations = json.load(f)
            #     print('  Object Segmentation: ', type(segmentations))
            #     print(segmentations.keys()) # a dictionary of 'params', 'sceneId', 'segIndices'
            #     segIndices = np.array(segmentations['segIndices'])
            #     # print(segIndices)       # assign a segment id to each point/vertex in the low res mesh
            #     # print(segIndices.shape) # same size as the number of points/vertices in low resolution mesh
            #     # Get objects classes
            #     with open(join(self.ply_path, scene, scene + '_vh_clean.aggregation.json'), 'r') as f:
            #         aggregation = json.load(f)
            #     print('  Object Classes: ', type(aggregation))
            #     print(aggregation.keys())   # a dictionary of 'sceneId', 'appId', 'segGroups', 'segmentsFile'.
            #     # Loop on object to classify points
            #     for segGroup in aggregation['segGroups']:
            #         print(segGroup)
            #         c_name = segGroup['label']
            #         if c_name in names1:
            #             nyuID = annot_to_nyuID[c_name]
            #             if nyuID in self.label_values:
            #                 for segment in segGroup['segments']:
            #                     vertices_labels[segIndices == segment] = nyuID
            #     # Save mesh
            #     # write_ply(join(mesh_path, scene + '_mesh.ply'),
            #     #           [vertices, vertices_colors, vertices_labels],
            #     #           ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
            #     write_ply(join(self.finer_pc_path, scene + '_test.ply'),
            #                 [sub_points, sub_colors, sub_labels, sub_vert_inds],
            #                 ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])
            # ###################

            ###########################
            # Create finer point clouds
            ###########################

            # Rasterize mesh with 3d points (place more point than enough to subsample them afterwards)
            points, associated_vert_inds = rasterize_mesh(vertices, faces, 0.003)

            # Subsample points
            sub_points, sub_vert_inds = grid_subsampling(points, labels=associated_vert_inds, sampleDl=0.01)

            # Collect colors from associated vertex
            sub_colors = vertices_colors[sub_vert_inds.ravel(), :]
            # print('finer point cloud')
            # print(vertices.shape)               # (128073, 3)
            # print(points.shape)                 # (12445079, 3)
            # print(associated_vert_inds.shape)   # (12445079,)
            # print(np.max(associated_vert_inds)) # 12807
            # print(np.min(associated_vert_inds)) # 0
            # print(sub_points.shape)             # (534865, 3)
            # print(sub_vert_inds.shape)          # (534865, 1)
            # print(sub_vert_inds.ravel().shape)  # (534865,)
            # print(np.max(sub_vert_inds.ravel()))# 12807
            # print(np.min(sub_vert_inds.ravel()))# 0
            # print(sub_colors.shape)             # (534865, 3)
            
            if self.set == 'test':
                # Save points
                write_ply(join(self.finer_pc_path, scene + '.ply'),
                            [sub_points, sub_colors, sub_vert_inds],
                            ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])
            else:
                # Collect labels from associated vertex
                sub_labels = vertices_labels[sub_vert_inds.ravel()]
                # Save points
                write_ply(join(self.finer_pc_path, scene + '.ply'),
                            [sub_points, sub_colors, sub_labels, sub_vert_inds],
                            ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])

            #  Display
            print('{:s} {:.1f} sec  / {:.1f}%'.format(scene,
                                                        time.time() - t1,
                                                        100 * i / N))
        
        print('Done in {:.1f}s'.format(time.time() - t0))
        return

    def prepare_room_ply(self, cloud_name):
        print('\nPreparing single ply files')
        t0 = time.time()

        # Mapping from annot to NYU labels ID
        label_files = join(self.ply_path, 'scannetv2-labels.combined.tsv')
        with open(label_files, 'r') as f:
            lines = f.readlines()
            names1 = [line.split('\t')[1] for line in lines[1:]]
            IDs = [int(line.split('\t')[4]) for line in lines[1:]]
            annot_to_nyuID = {n: id for n, id in zip(names1, IDs)}

        # for subset, new_path, mesh_path in zip(self.clouds, new_paths, mesh_paths):
        # Create folder
        if not exists(self.finer_pc_path):
            makedirs(self.finer_pc_path)
        if not exists(self.mesh_path):
            makedirs(self.mesh_path)

        N = len(self.clouds)
        
        for i, scene in enumerate(self.clouds):
            print('  processing: ', scene)
            ##############
            # Load meshes 
            ##############

            # Check if file already done
            if exists(join(self.finer_pc_path, scene + '.ply')):
                # vertex_data, faces = read_ply(join(self.mesh_path, scene + '_mesh.ply'), triangular_mesh=True)
                # for one_label in vertex_data['class']:
                #     print(one_label)
                # print(vertex_data['class'])
                # break
                continue
            t1 = time.time()

            # Read mesh
            # Add additional class label in the mesh file for training
            vertex_data, faces = read_ply(join(self.ply_path, scene, cloud_name + '_mesh.ply'), triangular_mesh=True)
            vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
            vertices_colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T

            vertices_labels = np.zeros(vertices.shape[0], dtype=np.int32)
            if self.set == 'test':
                # Save mesh
                write_ply(join(self.mesh_path, scene + '_mesh.ply'),
                            [vertices, vertices_colors],
                            ['x', 'y', 'z', 'red', 'green', 'blue'],
                            triangular_faces=faces)
            else:
                raise ValueError('TEST ONLY !!!!!!!')

            ###########################
            # Create finer point clouds
            ###########################

            # Rasterize mesh with 3d points (place more point than enough to subsample them afterwards)
            points, associated_vert_inds = rasterize_mesh(vertices, faces, 0.003)

            # Subsample points
            sub_points, sub_vert_inds = grid_subsampling(points, labels=associated_vert_inds, sampleDl=0.01)

            # Collect colors from associated vertex
            sub_colors = vertices_colors[sub_vert_inds.ravel(), :]

            if self.set == 'test':
                # Save points
                write_ply(join(self.finer_pc_path, scene + '.ply'),
                            [sub_points, sub_colors, sub_vert_inds],
                            ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])
            else:
                raise ValueError('TEST ONLY !!!!!!!')

            #  Display
            print('{:s} {:.1f} sec  / {:.1f}%'.format(scene,
                                                        time.time() - t1,
                                                        100 * i / N))
        
        print('Done in {:.1f}s'.format(time.time() - t0))
        return

    def load_subsampled_clouds(self):

        # Parameter
        # size of the first subsampling grid in meters, usually 0.04 here
        dl = self.config.first_subsampling_dl

        # Create path for files
        tree_path = join(self.path, 'scans/input_{:.3f}'.format(dl))
        if not exists(tree_path):
            makedirs(tree_path)

        ##############
        # Load KDTrees
        ##############
        
        # loop through all ply files previously created
        for i, file_path in enumerate(self.files):
            # Restart timer
            t0 = time.time()

            # Get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]

            # Name of the input files
            # KD-tree file
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            # subsampled point cloud file
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if exists(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                sub_vert_inds = data['vert_ind']
                if self.set == 'test':
                    sub_labels = None
                else:
                    sub_labels = data['class']

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                if self.set == 'test':
                    int_features = data['vert_ind']
                else:
                    int_features = np.vstack(
                        (data['vert_ind'], data['class'])
                    ).T

                # Subsample cloud
                sub_points, sub_colors, sub_int_features = grid_subsampling(points,
                                                                      features=colors,
                                                                      labels=int_features,
                                                                      sampleDl=dl)

                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255
                if self.set == 'test':
                    sub_vert_inds = np.squeeze(sub_int_features) # eg. from shape (1, 3, 1) to shape (3,), axis to be squeezed out must have size 1
                    sub_labels = None
                else:
                    sub_vert_inds = sub_int_features[:, 0]
                    sub_labels = sub_int_features[:, 1]

                # Get chosen neighborhoods
                # create KD-Tree for the subsampled point cloud, using sklearn package
                # class sklearn.neighbors.KDTree(array_like of shape (n_samples, n_features), positive int, ...)
                # looks like it's using KDTree from scipy.spatial.KDTree (same input format), sice it has data attributes while sklearn does not 
                # search_tree = KDTree(sub_points, leaf_size=50)    # tf uses 50, torch for s3dis uses 10
                search_tree = KDTree(sub_points, leaf_size=10)
                
                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                if self.set == 'test':
                    write_ply(sub_ply_file,
                              [sub_points, sub_colors, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])
                else:
                    write_ply(sub_ply_file,
                              [sub_points, sub_colors, sub_labels, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]
            self.input_vert_inds += [sub_vert_inds]
            if self.set in ['training', 'validation']:
                self.input_labels += [sub_labels]

            size = sub_colors.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time.time() - t0))            
            
        
        ############################
        # Coarse potential locations
        ############################
        # Only necessary for validation and test sets
        if self.use_potentials:
            print('\nPreparing potentials')
            
            # with 0.04 first_subsampling_dl, in_radius usually set to 2.0
            pot_dl = self.config.in_radius / 10
            cloud_ind = 0

            for i, file_path in enumerate(self.files):
                # Restart timer
                t0 = time.time()

                # get cloud name and split
                cloud_name = file_path.split('/')[-1][:-4]
                cloud_folder = file_path.split('/')[-2]

                coarse_KDTree_file = join(tree_path, '{:s}_coarse_KDTree.pkl'.format(cloud_name))
                if exists(coarse_KDTree_file):
                    # Read pkl with search tree
                    with open(coarse_KDTree_file, 'rb') as f:
                        search_tree = pickle.load(f)
                else:
                    # Subsample cloud
                    # KDTree.data returns an ndarray of shape (n, m), with n data points of dimension m to be indexed.
                    # this ndarry is the points used to create this KDTree.
                    # the array is not copied unless this is necessary
                    sub_points = np.array(self.input_trees[cloud_ind].data, copy=False)
                    # subsample it again to a coarser grid
                    coarse_points = grid_subsampling(sub_points.astype(np.float32), sampleDl=pot_dl)

                    # Get chosen neighborhoods
                    # search_tree = KDTree(coarse_points, leaf_size=50)
                    search_tree = KDTree(coarse_points, leaf_size=10)

                    # Save KDTree
                    with open(coarse_KDTree_file, 'wb') as f:
                        pickle.dump(search_tree, f)

                self.pot_trees += [search_tree]
                cloud_ind += 1

            print('Done in {:.1f}s'.format(time.time() - t0))


        ######################
        # Reprojection indices
        ######################
        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.set in ['validation', 'test', 'visualise']:

            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices
            # loop through original data points
            for i, file_path in enumerate(self.files):

                # Restart timer
                t0 = time.time()

                # get cloud name and split
                cloud_name = file_path.split('/')[-1][:-4]
                cloud_folder = file_path.split('/')[-2]

                # File name for saving
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))

                # Try to load previous indices
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    # data = read_ply(file_path)
                    # points = np.vstack((data['x'], data['y'], data['z'])).T
                    # labels = data['class']
                    # Get original mesh
                    vertex_data, faces = read_ply(join(self.mesh_path, cloud_name+'_mesh.ply'), triangular_mesh=True)
                    points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                    if self.set =='test':
                        labels = np.zeros(points.shape[0])
                    else:
                        labels = vertex_data['class']
                    print(points.shape)
                    print(labels.shape)

                    # Compute projection inds
                    # get the indices of the cloest neighbour for each input point
                    idxs = self.input_trees[i].query(points, return_distance=False)
                    #dists, idxs = self.input_trees[i_cloud].kneighbors(points)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print()
        return

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data, faces = read_ply(file_path, triangular_mesh=True)
        # data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/

# torch.utils.data.Sampler classes are used to specify 
# the sequence of indices/keys used in data loading. 
# They represent iterable objects over the indices to datasets.
# here is a custom Sampler object that at each time yields the next index/key to fetch.
class ScannetSampler(Sampler):
    """Sampler for Scannet"""
    # Every Sampler subclass has to provide an __iter__() method
    # and a __len__() method
    def __init__(self, dataset: ScannetDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step/forward-pass per epoch
        if dataset.set == 'training':
            self.N = dataset.config.epoch_steps
        else:
            self.N = dataset.config.validation_size

        return
    
    # __iter__() provides a way to iterate over 
    # indices of dataset elements 
    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """

        if not self.dataset.use_potentials:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0

            # Initiate container for indices
            all_epoch_inds = np.zeros((2, 0), dtype=np.int32)

            # Number of sphere centers taken per class in each cloud
            num_centers = self.N * self.dataset.config.batch_num
            random_pick_n = int(np.ceil(num_centers / (self.dataset.num_clouds * self.dataset.config.num_classes)))

            # Choose random points of each class for each cloud
            for cloud_ind, cloud_labels in enumerate(self.dataset.input_labels):
                epoch_indices = np.empty((0,), dtype=np.int32)
                for label_ind, label in enumerate(self.dataset.label_values):
                    if label not in self.dataset.ignored_labels:
                        label_indices = np.where(np.equal(cloud_labels, label))[0]
                        if len(label_indices) <= random_pick_n:
                            epoch_indices = np.hstack((epoch_indices, label_indices))
                        elif len(label_indices) < 50 * random_pick_n:
                            new_randoms = np.random.choice(label_indices, size=random_pick_n, replace=False)
                            epoch_indices = np.hstack((epoch_indices, new_randoms.astype(np.int32)))
                        else:
                            rand_inds = []
                            while len(rand_inds) < random_pick_n:
                                rand_inds = np.unique(np.random.choice(label_indices, size=5 * random_pick_n, replace=True))
                            epoch_indices = np.hstack((epoch_indices, rand_inds[:random_pick_n].astype(np.int32)))

                # Stack those indices with the cloud index
                epoch_indices = np.vstack((np.full(epoch_indices.shape, cloud_ind, dtype=np.int32), epoch_indices))

                # Update the global indice container
                all_epoch_inds = np.hstack((all_epoch_inds, epoch_indices))

            # Random permutation of the indices
            random_order = np.random.permutation(all_epoch_inds.shape[1])
            all_epoch_inds = all_epoch_inds[:, random_order].astype(np.int64)

            # Update epoch inds
            self.dataset.epoch_inds += torch.from_numpy(all_epoch_inds[:, :num_centers])

        # Generator loop
        for i in range(self.N):
            yield i

    # __len__() returns the length of the returned iterators.
    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N

    def fast_calib(self):
        """
        This method calibrates the batch sizes while ensuring the potentials are well initialized. Indeed on a dataset
        like Semantic3D, before potential have been updated over the dataset, there are cahnces that all the dense area
        are picked in the begining and in the end, we will have very large batch of small point clouds
        :return:
        """

        # Estimated average batch size and target value
        estim_b = 0
        target_b = self.dataset.config.batch_num

        # Calibration parameters
        low_pass_T = 10
        Kp = 100.0
        finer = False
        breaking = False

        # Convergence parameters
        smooth_errors = []
        converge_threshold = 0.1

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(2)

        for epoch in range(10):
            for i, test in enumerate(self):

                # New time
                t = t[-1:]
                t += [time.time()]

                # batch length
                b = len(test)

                # Update estim_b (low pass filter)
                estim_b += (b - estim_b) / low_pass_T

                # Estimate error (noisy)
                error = target_b - b

                # Save smooth errors for convergene check
                smooth_errors.append(target_b - estim_b)
                if len(smooth_errors) > 10:
                    smooth_errors = smooth_errors[1:]

                # Update batch limit with P controller
                self.dataset.batch_limit += Kp * error

                # finer low pass filter when closing in
                if not finer and np.abs(estim_b - target_b) < 1:
                    low_pass_T = 100
                    finer = True

                # Convergence
                if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                    breaking = True
                    break

                # Average timing
                t += [time.time()]
                mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d},  //  {:.1f}ms {:.1f}ms'
                    print(message.format(i,
                                         estim_b,
                                         int(self.dataset.batch_limit),
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1]))

            if breaking:
                break

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        """
        Method performing batch and neighbors calibration.
        Batch calibration:     Set "batch_limit" (the maximum number of points allowed in every batch) so that the
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
        if self.dataset.use_potentials:
            sampler_method = 'potentials'
        else:
            sampler_method = 'random'
        # key = 'potentials_2.000_0.040_1' -> test
        # key = 'potentials_2.000_0.040_4' -> train
        key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                               self.dataset.config.in_radius,
                                               self.dataset.config.first_subsampling_dl,
                                               self.dataset.config.batch_num)
        # print(key)
        # print(batch_lim_dict.keys())
        # print(batch_lim_dict.items())   # configuration key: current batch_limit
        
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
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
        # print(self.dataset.config.deform_layers)
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            # print('dl for layer', layer_ind, '=', dl)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            # print(key)
            # print(neighb_lim_dict.keys())
            # print(neighb_lim_dict.items())   # configuration key: neighb_limit in current layer

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
                key = '{:.3f}_{:.3f}'.format(dl, r)

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
            # get the volume of the ball: (4/3)*pi*(r+1)^3
            # deform_radius here is the number of grid, 
            # hence the volume is the total number of grid in the neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.deform_radius + 1) ** 3))
            # # debug
            # print(self.dataset.config.deform_radius)
            # print(self.dataset.config.deform_radius+1)
            # print(4 / 3 * np.pi * (self.dataset.config.deform_radius + 1) ** 3)
            print('volume in int:', hist_n)     # eg. 1437


            # Histogram of neighborhood sizes
            # initialise with zeros
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)
            print('neighbor hist shape:', neighb_hists.shape)   # (5, 1437)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config.batch_num

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            #####################
            # Perform calibration
            #####################
            # about the 2 loops here:
            # epoch: same as training, loop through all points for 10 times.
            # enumerate(dataloader): get batched data, loop till all points loaded or maximum steps reached
            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):
                    # print('  ', epoch, '-', batch_i)
                    # print(type(batch))  # each batch is a ScannetCustomBatch object defined below
                    # print(batch)
                    # print(batch.neighbors)

                    # Update neighborhood histogram
                    # count the number of neighbors with idx < shape[0](num of pts?)
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    # print('')
                    # print(hists)
                    neighb_hists += np.vstack(hists)

                    # batch length  # initialised with 1
                    b = len(batch.cloud_inds)
                    # print('  ', 'b =', b)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T
                    # print('  ', 'estim_b =', estim_b, 'lowpass =', low_pass_T)

                    # Estimate error (noisy)
                    error = target_b - b
                    # print('  ', 'error =', error)
                    # print('  ', 'smooth error =', target_b - estim_b)

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller (maximum number of points per batch)
                    self.dataset.batch_limit += Kp * error

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
                                             int(self.dataset.batch_limit)))
                    
                    # print(batch_i, 'finished')

                if breaking:
                    break
            # print('both loop finished')

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            # print(cumsum)
            # print(cumsum.shape)           # (1473, 5)
            # print(cumsum[hist_n - 1, :])  # get last row, 
                                            # i.e. total number of points inside the sphere
            # count the number of neighbors with the untouched ratio applied
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles
            # print(percentiles)    # [38, 37, 228, 173, 71], up to which row reaches the limit

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

            # Save batch_limit dictionary
            if self.dataset.use_potentials:
                sampler_method = 'potentials'
            else:
                sampler_method = 'random'
            key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                                   self.dataset.config.in_radius,
                                                   self.dataset.config.first_subsampling_dl,
                                                   self.dataset.config.batch_num)
            batch_lim_dict[key] = float(self.dataset.batch_limit)
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)


        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class ScannetCustomBatch:
    """Custom batch definition with memory pinning for Scannet"""

    def __init__(self, input_list):
        # input list is a list of batches
        # print('loading batches:', len(input_list))  # 1 batch only
        
        # Get rid of batch dimension
        input_list = input_list[0]
        # print(input_list)
        # print(len(input_list))  # list of 32 ndarray
        # thus here only one batch of data is kept

        # Number of layers
        L = (len(input_list) - 7) // 5  # // is floor division, 
        # print(L)  # L = 5

        #### Example input list see example_batch_data.txt in dataset folder
        # Extract input tensors from the list of numpy array
        ind = 0
        # points (coordinates) in each layer
        # eg. [21604, 3], [5473, 3], [1392, 3], [335, 3], [86, 3]
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        # print(len(self.points))
        # for pts in self.points:
        #     print(pts.size())
        
        ind += L
        # neighbors indices
        # eg. [21604, 63], [5473, 61], [1392, 397], [335, 277], [86, 86]
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        # print(len(self.neighbors))
        # for pts in self.neighbors:
        #     print(pts.size())
        
        ind += L
        # pooling indices
        # eg. [5473, 58], [1392, 57], [335, 394], [86, 273], [0, 1]
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        # print(len(self.pools))
        # for pts in self.pools:
        #     print(pts.size())
        
        ind += L
        # upsampling indices
        # eg. [21604, 62], [5473, 60], [1392, 281], [335, 86], [0, 1]
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        # print(len(self.upsamples))
        # for pts in self.upsamples:
        #     print(pts.size())
        
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        
        ind += L
        # dim = [21604, 4], val = [1, r, g, b]
        self.features = torch.from_numpy(input_list[ind])
        # print(self.features.size())
        
        ind += 1
        # [21604]
        self.labels = torch.from_numpy(input_list[ind])
        # print(self.labels.size())
        
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        # print(self.scales.size())

        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        # print(self.rots.size())

        ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind])
        # print(self.cloud_inds.size())

        ind += 1
        # Index of the center point?
        self.center_inds = torch.from_numpy(input_list[ind])
        # print('center point indices', self.center_inds)

        ind += 1
        # [21604]
        self.input_inds = torch.from_numpy(input_list[ind])
        # print(self.input_inds.size())

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
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()

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
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)

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

# After fetching a list of samples using the indices from sampler
# this function is passed as the collate_fn argument and is used 
# to collate lists of samples into batches.
# In this case, loading from a map-style dataset is roughly 
# equivalent with:
#         for indices in batch_sampler:
#             yield collate_fn([dataset[i] for i in indices])
# More about yield: return from a function without destroying the 
# states of its local variables. i.e. get returned value from the 
# function and continue execute the function.
def ScannetCollate(batch_data):
    return ScannetCustomBatch(batch_data)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Debug functions
#       \*********************/


def debug_upsampling(dataset, loader):
    """Shows which labels are sampled according to strategy chosen"""


    for epoch in range(10):

        for batch_i, batch in enumerate(loader):

            pc1 = batch.points[1].numpy()
            pc2 = batch.points[2].numpy()
            up1 = batch.upsamples[1].numpy()

            print(pc1.shape, '=>', pc2.shape)
            print(up1.shape, np.max(up1))

            pc2 = np.vstack((pc2, np.zeros_like(pc2[:1, :])))

            # Get neighbors distance
            p0 = pc1[10, :]
            neighbs0 = up1[10, :]
            neighbs0 = pc2[neighbs0, :] - p0
            d2 = np.sum(neighbs0 ** 2, axis=1)

            print(neighbs0.shape)
            print(neighbs0[:5])
            print(d2[:5])

            print('******************')
        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_timing(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.config.batch_num
    estim_N = 0

    #####################################################################
    # ERROR: Unexpected floating-point exception encountered in worker. #
    #####################################################################
    for epoch in range(10):

        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.cloud_inds) - estim_b) / 100
            estim_N += (batch.features.shape[0] - estim_N) / 10

            # Pause simulating computations
            time.sleep(0.05)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                message = 'Step {:02d}-{:04d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f} - {:.0f}'
                print(message.format(epoch, batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1],
                                     estim_b,
                                     estim_N))

        print('************* Epoch ended *************')

    # print(type(dataset.input_labels))
    # print(len(dataset.input_labels))

    # bug with the count here
    # print(dataset.input_labels) list of arrays
    # print(type(dataset.input_labels))
    # print(dataset.input_labels.shape)
    for input_label in dataset.input_labels:
        _, counts = np.unique(input_label, return_counts=True)
        print(counts)


def debug_show_clouds(dataset, loader):


    for epoch in range(10):

        clouds = []
        cloud_normals = []
        cloud_labels = []

        L = dataset.config.num_layers

        for batch_i, batch in enumerate(loader):

            # Print characteristics of input tensors
            print('\nPoints tensors')
            for i in range(L):
                print(batch.points[i].dtype, batch.points[i].shape)
            print('\nNeigbors tensors')
            for i in range(L):
                print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
            print('\nPools tensors')
            for i in range(L):
                print(batch.pools[i].dtype, batch.pools[i].shape)
            print('\nStack lengths')
            for i in range(L):
                print(batch.lengths[i].dtype, batch.lengths[i].shape)
            print('\nFeatures')
            print(batch.features.dtype, batch.features.shape)
            print('\nLabels')
            print(batch.labels.dtype, batch.labels.shape)
            print('\nAugment Scales')
            print(batch.scales.dtype, batch.scales.shape)
            print('\nAugment Rotations')
            print(batch.rots.dtype, batch.rots.shape)
            print('\nModel indices')
            print(batch.model_inds.dtype, batch.model_inds.shape)

            print('\nAre input tensors pinned')
            print(batch.neighbors[0].is_pinned())
            print(batch.neighbors[-1].is_pinned())
            print(batch.points[0].is_pinned())
            print(batch.points[-1].is_pinned())
            print(batch.labels.is_pinned())
            print(batch.scales.is_pinned())
            print(batch.rots.is_pinned())
            print(batch.model_inds.is_pinned())

            show_input_batch(batch)

        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_batch_and_neighbors_calib(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)

    for epoch in range(10):

        for batch_i, input_list in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Pause simulating computations
            time.sleep(0.01)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> Average timings (ms/batch) {:8.2f} {:8.2f} '
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1]))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)

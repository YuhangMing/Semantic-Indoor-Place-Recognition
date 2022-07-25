#
#      Performing Semantic Segmentation Test
#      Yuhang Modifidation
#


# Common libs
from operator import imod
import signal
import os
import numpy as np
import sys
import torch

# Dataset
# cloud segmentation using rgbd pcd from 7 scenes
from datasets.SinglePLY import *
# SLAM segmentation
from datasets.ScannetSLAM import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN

# Visualisation
import open3d as o3d
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def model_choice(chosen_log):

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])
        # print(test_dataset)

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
        # print(logs)

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break
        # print(chosen_log)

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    # chosen_log = 'results/Log_2021-06-16_02-31-04'  # => ScanNetSLAM (full), batch 8, 1st feat 64, 0.04-2.0, without color
    chosen_log = 'results/Log_2021-06-16_02-42-30'  # => ScanNetSLAM (full), batch 8, 1st feat 64, 0.04-2.0, with color
        
    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = 0 # chkp_500
    # chkp_idx = 9 # chkp_500

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    config.batch_num = 1
    config.val_batch_num = 1
    #config.in_radius = 4
    config.validation_size = 300    # decide how many points will be covered in prediction
    config.input_threads = 0
    config.max_in_points = 9000
    config.max_val_points = 9000
    config.print_current()

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')
    # Initiate dataset
    # Use the provided dataset and loader, for easy batch generation
    #### ScannetSLAM
    test_dataset = ScannetSLAMDataset(config, 'validation', balance_classes=False)
    # test_dataset = ScannetSLAMDataset(config, 'test', balance_classes=False)
    test_sampler = ScannetSLAMSampler(test_dataset)
    collate_fn = ScannetSLAMCollate

    print(test_dataset.label_values)
    print(test_dataset.ignored_labels)
    
    # Data loader with automatic batching enabled
    # https://pytorch.org/docs/stable/data.html
    # torch.utils.data.DataLoader default parameters
    # DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
    #        batch_sampler=None, num_workers=0, collate_fn=None,
    #        pin_memory=False, drop_last=False, timeout=0,
    #        worker_init_fn=None, *, prefetch_factor=2,
    #        persistent_workers=False)
    # collate_fn: -> Create batched input
    #   collate_fn is called with a list of data samples at each time. 
    #   It is expected to collate the input samples into a batch for 
    #   yielding from the data loader iterator.
    #   For instance, if each data sample consists of a 3-channel image 
    #   and an integral class label, i.e., each element of the dataset 
    #   returns a tuple (image, class_index), the default collate_fn 
    #   collates a list of such tuples into a single tuple of a batched 
    #   image tensor and a batched class label Tensor 
    #   (4D_tensor_as_batched_images, 1D_tensor_as_batched_labels). 
    #   In particular, the default collate_fn has the following properties:
    #   - It always prepends a new dimension as the batch dimension.
    #   - It automatically converts NumPy arrays and Python numerical 
    #     values into PyTorch Tensors.
    #   - It preserves the data structure, e.g., if each sample is a 
    #     dictionary, it outputs a dictionary with the same set of keys 
    #     but batched Tensors as values (or lists if the values can not 
    #     be converted into Tensors). Same for lists, tuples, namedtuples, etc.
    # num_works: -> get data in parallel
    #   Setting the argument num_workers as a positive integer will turn on multi-
    #   process data loading with the specified number of loader worker processes.
    #   each time an iterator of a DataLoader is created (e.g., when you call 
    #   enumerate(dataloader)), num_workers worker processes are created. At this 
    #   point, the dataset, collate_fn, and worker_init_fn are passed to each 
    #   worker, where they are used to initialize, and fetch data. This means that 
    #   dataset access together with its internal IO, transforms (including 
    #   collate_fn) runs in the worker process.
    #   For map-style datasets, the main process generates the indices using  
    #   sampler and sends them to the workers. So any shuffle randomization is  
    #   done in the main process which guides loading by assigning indices to load
    # pin_memory: -> faster data transfer
    #   put the fetched data Tensors in pinned memory, and thus enables faster 
    #   data transfer to CUDA-enabled GPUs.
    #   pin_memory needs to be specifically defined if using custom batch
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)
    print('Calibed batch limit:', test_sampler.dataset.batch_limit)
    print('Calibed neighbor limit:', test_sampler.dataset.neighborhood_limits)

    print('\nModel Preparation')
    print('*****************')
    # Define network model
    t1 = time.time()
    if config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********')
    # Perform prediction
    if config.dataset_task == 'slam_segmentation':
        tester.slam_segmentation_test(net, test_loader, config, 0)
    else:
        raise ValueError('Unsupported dataset task: ' + config.dataset_task)

    # # Visualisation, uncomment below if needed
    # # plot legends
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.axis([0, 4, 0, 20])
    # for i in range(13):
    #     y = 20 - (i*3+1)
    #     if y > 0:
    #         x = 0.5
    #     else:
    #         x = 2.5
    #         y = -y
    #     ax.plot([x], [y], '.', color=(test_dataset.label_to_colour[i][0]/255.0, 
    #                                 test_dataset.label_to_colour[i][1]/255.0, 
    #                                 test_dataset.label_to_colour[i][2]/255.0),  
    #             markersize=40) 
    #     ax.text(x+0.25, y-0.5, test_dataset.label_to_names[i], fontsize=15)
    # # plot color annotation
    # plt.show()


#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#      Yuhang Modifidation
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
import numpy as np
import sys
import torch

# Dataset
from datasets.ModelNet40 import *
from datasets.S3DIS import *
from datasets.SemanticKitti import *
from datasets.Scannet import *
from datasets.SinglePLY import *
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

    #########################
    # Choose the model to use
    #########################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    # chosen_log = 'last_S3DIS'
    # chosen_log = 'results/Log_2021-04-19_10-01-23'  # => S3DIS, batch 4, 1st feat 64, 0.04-2.0
    chosen_log = 'results/Log_2021-05-05_06-19-46'  # => ScanNet (subset), batch 10, 1st feat 64, 0.04-2.0
    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None
    

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
    # print(chkps)
    # print(np.sort(chkps)) # sort string in alphbatic order

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)
    config.print_current()

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    config.batch_num = 1
    #config.in_radius = 4
    config.validation_size = 50    # decide how many points will be covered in prediction -> how many forward passes
    # 50 is a suitable value to cover a room-scale point cloud
    # 4 is a suitable value to cover a rgbd slam input size point cloud
    config.input_threads = 10
    config.print_current()

    ##############
    # Prepare Data
    ##############

    print('\nData Preparation')
    print('****************')
    # Initiate dataset
    # Use the provided dataset and loader, for easy batch generation
    # test_dataset = S3DISDataset(config, 'visualise', use_potentials=True)
    # test_sampler = S3DISSampler(test_dataset)
    # collate_fn = S3DISCollate
    # test_dataset = SinglePlyDataset(config, 'visualise', use_potentials=True)
    # test_sampler = SinglePlySampler(test_dataset)
    # collate_fn = SinglePlyCollate
    # test_dataset = ScannetDataset(config, 'validation', use_potentials=True)
    test_dataset = ScannetDataset(config, 'test', use_potentials=True)
    test_sampler = ScannetSampler(test_dataset)
    collate_fn = ScannetCollate
    print(test_dataset.label_values)
    print(test_dataset.ignored_labels)
    
    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)
    
    print('\nModel Preparation')
    print('*****************')
    # load network architecture
    t1 = time.time()
    if config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))


    print('\nTest with different blocks')
    print('**************************')
    # for child in net.children():
    #     print(child, '\n')
    # for mod in net.modules():
    #     print(mod, '\n')
    # get the encoder out    
    for name, module in net.named_children():
        if 'encoder' in name:
            encoder = module
    # print(encoder)
    for child in encoder.children():
        print(child, '\n')

    print('\nStart test')
    print('**********\n')
    # Perform prediction
    tester.segmentation_with_return(net, test_loader, config, 0)
    # # Forward pass
    # outputs = net(batch, config)
    # inter_en_feat = net.inter_encoder_features(batch, config)

    # Visualisation
    print('\nVisualising Predictions')
    print('***********************\n')
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # y_lim = test_dataset.num_classes *2
    # ax.axis([0, 4, 0, y_lim])
    # for i, label in enumerate(test_dataset.label_values):
    #     y = y_lim - (i*3+1)
    #     if y > 0:
    #         x = 0.5
    #     else:
    #         x = 2.5
    #         y = -y
    #     ax.plot([x], [y], '.', color=(test_dataset.colour_to_label[label][0]/255.0, 
    #                                 test_dataset.colour_to_label[label][1]/255.0, 
    #                                 test_dataset.colour_to_label[label][2]/255.0),  
    #             markersize=40) 
    #     ax.text(x+0.25, y-0.5, test_dataset.label_to_names[label], fontsize=15)
    # # plot color annotation
    # plt.show()

    # # S3DIS
    # rgb = o3d.io.read_point_cloud(join(test_dataset.path, 
    #                                    test_dataset.train_path, 
    #                                    'Area_5', test_dataset.room_name + ".ply"))
    # pred = o3d.io.read_point_cloud(join('test', 
    #                                     config.saving_path.split('/')[-1], 
    #                                     'predictions',
    #                                     test_dataset.room_name+'_pred.ply'))
    # Scannet
    room_name = test_dataset.clouds[0]
    rgb = o3d.io.read_point_cloud(join(test_dataset.ply_path,
                                       room_name,
                                       room_name+'_vh_clean_2.ply'))
    pred = o3d.io.read_point_cloud(join('test', 
                                        config.saving_path.split('/')[-1], 
                                        'predictions',
                                        room_name+'_pred.ply'))
    # # 7Scenes
    # room_name = test_dataset.clouds[0]
    # rgb = o3d.io.read_point_cloud(join(test_dataset.ply_path,
    #                                    room_name,
    #                                    'point_cloud_620_mesh.ply'))
    # pred = o3d.io.read_point_cloud(join('test', 
    #                                     config.saving_path.split('/')[-1], 
    #                                     'predictions',
    #                                     room_name+'_pred.ply'))

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='RGB', width=960, height=540, left=0, top=0)
    vis1.add_geometry(rgb)
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='prediction', width=960, height=540, left=0, top=0)
    vis2.add_geometry(pred)
    # visualise segmentation result with original color pc
    while True:
        vis1.update_geometry(rgb)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(pred)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()

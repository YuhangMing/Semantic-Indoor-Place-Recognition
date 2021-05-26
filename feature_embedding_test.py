#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      |    Semantic VLAD Recognition    |
#      0=================================0
#
#      Yuhang 
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import os
# import sys
import signal
import numpy as np
import torch

# Dataset
from torch.utils.data import DataLoader
# stanford cloud segmentation
# from datasets.S3DIS import *
# scannet cloud segmentation
# from datasets.Scannet import *
# cloud segmentation using rgbd pcd from 7 scenes
# from datasets.SinglePLY import *
# SLAM segmentation
# from datasets.ScannetSLAM import *
# vlad
from datasets.ScannetTriple import *

from models.architectures import KPFCNN
from models.PRNet import PRNet
from utils.config import Config
from utils.trainer import RecogModelTrainer
# from utils.tester import ModelTester

# VLAD
from sklearn.neighbors import KDTree

# # Visualisation
# import open3d as o3d
# import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#
#
#   use single GPU, use 
#   export CUDA_VISIBLE_DEVICES=3
#   in terminal
#

bTRAIN = False

if __name__ == '__main__':

    ######################
    # LOAD THE PRE-TRAINED 
    # SEGMENTATION NETWORK
    ######################

    print('\nLoad pre-trained segmentation KP-FCNN')
    print('*************************************')
    t = time.time()
    # chosen_log = 'results/Log_2021-05-05_06-19-46'  # => ScanNet (subset), batch 10, 1st feat 64, 0.04-2.0
    chosen_log = 'results/Log_2021-05-14_02-21-27'  # => ScanNetSLAM (subset), batch 8, 1st feat 64, 0.04-2.0
    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None
    print('Chosen log:', chosen_log, 'chkp_idx=', chkp_idx)

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']
    print('Checkpoints found:', chkps)
    # print(np.sort(chkps)) # sort string in alphbatic order

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)
    print('Checkpoints chosen:', chosen_chkp)

    # Initialise and Load the configs
    config = Config()
    config.load(chosen_log)
    # Change parameters for the TESTing here. 
    # For example, you can stop augmenting the input data.
    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    config.batch_num = 1        # for cloud segmentation
    config.val_batch_num = 1    # for SLAM segmentation
    #config.in_radius = 4
    config.validation_size = 50    # decide how many points will be covered in prediction -> how many forward passes
    # 50 is a suitable value to cover a room-scale point cloud
    # 4 is a suitable value to cover a rgbd slam input size point cloud
    config.input_threads = 0
    config.print_current()

    # set label manually here for scannet segmentation
    # with the purpose of putting loading parts together
    label_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    ignored_labels = [0]
    # Initialise segmentation network
    seg_net = KPFCNN(config, label_values, ignored_labels)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")        
    seg_net.to(device)
    # Load pretrained weights
    checkpoint = torch.load(chosen_chkp)
    # print(checkpoint.keys())    # ['epoch', 'model_state_dict', 'optimizer_state_dict', 'saving_path']
    # print(checkpoint['model_state_dict'].keys())    # where weights are stored
    # print(checkpoint['optimizer_state_dict'].keys())
    seg_net.load_state_dict(checkpoint['model_state_dict'])
    # number of epoch trained
    epoch = checkpoint['epoch']
    
    # set to evaluation mode
    # Dropout and BatchNorm (and maybe some custom modules) behave differently during training and 
    # evaluation. You must let the model know when to switch to eval mode by calling .eval() on 
    # the model. This sets self.training to False for every module in the model. 
    seg_net.eval()
    
    print("SEGMENTATION model and training state restored with", epoch, "epoches trained.")
    print('Done in {:.1f}s\n'.format(time.time() - t))

    if bTRAIN:

        ###########################
        # TRAIN RECOGNITION NETWORK
        ###########################

        print('\nData Preparation')
        print('****************')
        t = time.time()
        # new dataset for triplet input
        train_dataset = ScannetTripleDataset(config, 'training', balance_classes=False)
        # val_dataset = ScannetTripleDataset(config, 'validation', balance_classes=False)

        # Initialize samplers
        train_sampler = ScannetTripleSampler(train_dataset)
        # val_sampler = ScannetTripleSampler(val_dataset)

        # Initialize the dataloader
        train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler,
                                collate_fn=ScannetTripleCollate, num_workers=config.input_threads,
                                pin_memory=True)
        # val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler,
        #                          collate_fn=ScannetTripleCollate, num_workers=config.input_threads,
        #                          pin_memory=True)

        # Calibrate samplers
        train_sampler.calibration(train_loader, verbose=True)
        # val_sampler.calibration(val_loader, verbose=True)
        print('Calibed batch limit:', train_sampler.dataset.batch_limit)
        print('Calibed neighbor limit:', train_sampler.dataset.neighborhood_limits)
        print('Done in {:.1f}s\n'.format(time.time() - t))

        
        print('\nPrepare Recognition Model')
        print('*************************')
        reg_net = PRNet(config)
        # for k, v in reg_net.named_parameters():
        #     print(k, v)
        # print(reg_net.named_parameters())

        # Choose here if you want to start training from a previous snapshot (None for new training)
        # previous_training_path = 'Recog_Log_2021-05-24_14-15-04'
        previous_training_path = ''

        # Choose index of checkpoint to start from. If None, uses the latest chkp
        chkp_idx = None # override here
        if previous_training_path:
            # Find all snapshot in the chosen training folder
            chkp_path = os.path.join('results', previous_training_path, 'checkpoints') # override here
            chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp'] # override here
            # Find which snapshot to restore
            if chkp_idx is None:
                chosen_chkp = 'current_chkp.tar'
            else:
                chosen_chkp = np.sort(chkps)[chkp_idx]
            chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)
        else:
            chosen_chkp = None

        # update parameters for recog training
        config.max_epoch = 50
        config.checkpoint_gap = 10
        if config.saving:
            config.saving_path = time.strftime('results/Recog_Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        print('Updated max_epoch =', config.max_epoch)
        print('Updated checkpoint_gap =', config.checkpoint_gap)
        print('Updated saving_path =', config.saving_path)


        print('\nPrepare Trainer')
        print('***************')
        # initialise trainier
        trainer = RecogModelTrainer(reg_net, config)
        
        print('\nStart training')
        print('**************')
        # TRAINING
        trainer.train(reg_net, seg_net, train_loader, config)
        print('Forcing exit now')
        os.kill(os.getpid(), signal.SIGINT)

    else:

        ##########################
        # TEST RECOGNITION NETWORK
        ##########################
        
        print('\nLoad pre-trained recognition VLAD')
        print('*********************************')
        t = time.time()
        chosen_log = 'results/Recog_Log_2021-05-25_10-55-26'
        # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
        chkp_idx = None
        print('Chosen log:', chosen_log, 'chkp_idx=', chkp_idx)

        # Find all checkpoints in the chosen training folder
        chkp_path = os.path.join(chosen_log, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']
        print('Checkpoints found:', chkps)
        # print(np.sort(chkps)) # sort string in alphbatic order

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)
        print('Checkpoints chosen:', chosen_chkp)

        # Initialise and Load the configs
        config = Config()
        config.load(chosen_log)
        # # Change parameters for the TESTing here. 
        # config.batch_num = 1        # for cloud segmentation
        # config.val_batch_num = 1    # for SLAM segmentation
        # config.validation_size = 50    # decide how many points will be covered in prediction -> how many forward passes
        # config.input_threads = 0
        config.print_current()

        # Initialise segmentation network
        reg_net = PRNet(config)
        reg_net.to(device)

        # Load pretrained weights
        checkpoint = torch.load(chosen_chkp)
        reg_net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        reg_net.eval()

        print("RECOGNITION model and training state restored with", epoch, "epoches trained.")
        print('Done in {:.1f}s\n'.format(time.time() - t))


        print('\nData Preparation')
        print('****************')
        t = time.time()
        # new dataset for triplet input
        test_dataset = ScannetTripleDataset(config, 'test', balance_classes=False)
        test_sampler = ScannetTripleSampler(test_dataset)

        # Initialize the dataloader
        test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler,
                                collate_fn=ScannetTripleCollate, num_workers=config.input_threads,
                                pin_memory=True)

        # Calibrate samplers
        test_sampler.calibration(test_loader, verbose=True)
        print('Calibed batch limit:', test_sampler.dataset.batch_limit)
        print('Calibed neighbor limit:', test_sampler.dataset.neighborhood_limits)
        print('Done in {:.1f}s\n'.format(time.time() - t))


        print('\nGenerate database')
        print('*****************')
        t = time.time()
        # Get database
        break_cnt = 0
        map_database= []
        ind_dict = {}
        for i, batch in enumerate(test_loader):
            # continue if empty input list is given
            # caused by empty positive neighbors
            if len(batch.points) == 0:
                break_cnt +=1
                continue
            else:
                break_cnt = 0
            # stop fetch new batch if no more points left
            if break_cnt > 4:
                break

            if i%2 == 0:
                batch.to(device)
                # get the VLAD descriptor
                # list of interim vectors
                feat = seg_net.inter_encoder_features(batch)
                vlad = reg_net(feat)
                # print(type(vlad.cpu().detach().numpy()))
                # print(type(vlad.cpu().detach().numpy()[0]))
                # print(vlad.cpu().detach().numpy().shape)
                map_database.append(vlad.cpu().detach().numpy()[0]) # append a (1,256) np.ndarray
                ind_dict[int(i/2)] = i
            
        map_database = np.array(map_database)
        search_tree = KDTree(map_database, leaf_size=4)
        # print(map_database.shape)
        print('Done in {:.1f}s\n'.format(time.time() - t))

        print('\nStart test')
        print('**********')
        t = time.time()
        # loop again to test with KDTree NN
        break_cnt = 0
        test_pair = []
        for i, batch in enumerate(test_loader):
            # continue if empty input list is given
            # caused by empty positive neighbors
            if len(batch.points) == 0:
                break_cnt +=1
                continue
            else:
                break_cnt = 0
            # stop fetch new batch if no more points left
            if break_cnt > 4:
                break
            
            if i%2 != 0:
                batch.to(device)
                # get the VLAD descriptor
                # list of interim vectors
                feat = seg_net.inter_encoder_features(batch)
                vlad = reg_net(feat).cpu().detach().numpy()     # ndarray of (1, 256)
                dist, ind = search_tree.query(vlad)
                print(i, ind, ind_dict[ind[0][0]], dist)
                test_pair.append([i, ind_dict[ind[0][0]]])

            # ind = test_loader.dataset.epoch_inds[i]
            # s_ind, f_ind = test_loader.dataset.all_inds[ind]
            # current_file = test_loader.dataset.files[s_ind][f_ind]
            # print(i, ind, s_ind, f_ind, current_file)
        
        print('Done in {:.1f}s\n'.format(time.time() - t))


        print('\nVisualisation')
        print('*************')
        ## Add visualisation here


    #################################################################################
    # ^ train/test recog net
    #
    #################################################################################
    #
    # V test segmentation
    #################################################################################


    # # tester.intermedian_features(seg_net, train_loader, config)

    # # Initiate dataset
    # # Use the provided dataset and loader, for easy batch generation
    # #### S3DIS
    # # test_dataset = S3DISDataset(config, 'visualise', use_potentials=True)
    # # test_sampler = S3DISSampler(test_dataset)
    # # collate_fn = S3DISCollate
    # #### 7Scenes
    # # test_dataset = SinglePlyDataset(config, 'visualise', use_potentials=True)
    # # test_sampler = SinglePlySampler(test_dataset)
    # # collate_fn = SinglePlyCollate
    # #### Scannet
    # # # test_dataset = ScannetDataset(config, 'validation', use_potentials=True)
    # # test_dataset = ScannetDataset(config, 'test', use_potentials=True)
    # # test_sampler = ScannetSampler(test_dataset)
    # # collate_fn = ScannetCollate
    # #### ScannetSLAM
    # test_dataset = ScannetSLAMDataset(config, 'test', balance_classes=False)
    # test_sampler = ScannetSLAMSampler(test_dataset)
    # collate_fn = ScannetSLAMCollate
    
    # print(test_dataset.label_values)
    # print(test_dataset.ignored_labels)
    
    # # Data loader
    # test_loader = DataLoader(test_dataset,
    #                          batch_size=1,
    #                          sampler=test_sampler,
    #                          collate_fn=collate_fn,
    #                          num_workers=config.input_threads,
    #                          pin_memory=True)

    # # Calibrate samplers
    # test_sampler.calibration(test_loader, verbose=True)
    # print('Calibed batch limit:', test_sampler.dataset.batch_limit)
    # print('Calibed neighbor limit:', test_sampler.dataset.neighborhood_limits)
    
    # print('\nModel Preparation')
    # print('*****************')
    # # load network architecture
    # t1 = time.time()
    # if config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
    #     net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    # else:
    #     raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # # Define a visualizer class
    # tester = ModelTester(net, chkp_path=chosen_chkp)
    # print('Done in {:.1f}s\n'.format(time.time() - t1))

    # print('\nStart test')
    # print('**********')
    # # Perform prediction
    # if config.dataset_task == 'cloud_segmentaion':
    #     # tester.cloud_segmentation_test(net, test_loader, config, 0)
    #     tester.segmentation_with_return(net, test_loader, config, 0)
    # elif config.dataset_task == 'slam_segmentation':
    #     tester.slam_segmentation_test(net, test_loader, config, 0)
    # else:
    #     raise ValueError('Unsupported dataset task: ' + config.dataset_task)
    # # # Forward pass
    # # outputs = net(batch, config)
    # # inter_en_feat = net.inter_encoder_features(batch, config)

    # # # print('\nTest with different blocks')
    # # # print('**************************')
    # # # # for child in net.children():
    # # # #     print(child, '\n')
    # # # # for mod in net.modules():
    # # # #     print(mod, '\n')
    # # # # get the encoder out    
    # # # for name, module in net.named_children():
    # # #     if 'encoder' in name:
    # # #         encoder = module
    # # # # print(encoder)
    # # # for child in encoder.children():
    # # #     print(child, '\n')

    # # print('\nGet intermedian features')
    # # print('************************')

    # # tester.intermedian_features(net, test_loader, config)


    #################################################################################
    # # Visualisation
    # print('\nVisualising Predictions')
    # print('***********************\n')
    # # fig = plt.figure()
    # # ax = fig.add_subplot()
    # # y_lim = test_dataset.num_classes *2
    # # ax.axis([0, 4, 0, y_lim])
    # # for i, label in enumerate(test_dataset.label_values):
    # #     y = y_lim - (i*3+1)
    # #     if y > 0:
    # #         x = 0.5
    # #     else:
    # #         x = 2.5
    # #         y = -y
    # #     ax.plot([x], [y], '.', color=(test_dataset.label_to_colour[label][0]/255.0, 
    # #                                 test_dataset.label_to_colour[label][1]/255.0, 
    # #                                 test_dataset.label_to_colour[label][2]/255.0),  
    # #             markersize=40) 
    # #     ax.text(x+0.25, y-0.5, test_dataset.label_to_names[label], fontsize=15)
    # # # plot color annotation
    # # plt.show()

    # # # S3DIS
    # # rgb = o3d.io.read_point_cloud(join(test_dataset.path, 
    # #                                    test_dataset.train_path, 
    # #                                    'Area_5', test_dataset.room_name + ".ply"))
    # # pred = o3d.io.read_point_cloud(join('test', 
    # #                                     config.saving_path.split('/')[-1], 
    # #                                     'predictions',
    # #                                     test_dataset.room_name+'_pred.ply'))
    # # Scannet
    # room_name = test_dataset.clouds[0]
    # rgb = o3d.io.read_point_cloud(join(test_dataset.ply_path,
    #                                    room_name,
    #                                    room_name+'_vh_clean_2.ply'))
    # pred = o3d.io.read_point_cloud(join('test', 
    #                                     config.saving_path.split('/')[-1], 
    #                                     'predictions',
    #                                     room_name+'_pred.ply'))
    # # # 7Scenes
    # # room_name = test_dataset.clouds[0]
    # # rgb = o3d.io.read_point_cloud(join(test_dataset.ply_path,
    # #                                    room_name,
    # #                                    'point_cloud_620_mesh.ply'))
    # # pred = o3d.io.read_point_cloud(join('test', 
    # #                                     config.saving_path.split('/')[-1], 
    # #                                     'predictions',
    # #                                     room_name+'_pred.ply'))

    # ScannetSLAM
    # pred_path = '/NNs/Semantic-Global-Localisation/test/Log_2021-05-13_14-38-15/predictions'
    # rgb_path = '/media/yohann/My Passport/datasets/ScanNet/scans/input_pcd/scene0000_01/'
    # for file_name in listdir(pred_path):


    # vis1 = o3d.visualization.Visualizer()
    # vis1.create_window(window_name='RGB', width=960, height=540, left=0, top=0)
    # vis1.add_geometry(rgb)
    # vis2 = o3d.visualization.Visualizer()
    # vis2.create_window(window_name='prediction', width=960, height=540, left=0, top=0)
    # vis2.add_geometry(pred)
    # # visualise segmentation result with original color pc
    # while True:
    #     vis1.update_geometry(rgb)
    #     if not vis1.poll_events():
    #         break
    #     vis1.update_renderer()

    #     vis2.update_geometry(pred)
    #     if not vis2.poll_events():
    #         break
    #     vis2.update_renderer()

    # vis1.destroy_window()
    # vis2.destroy_window()

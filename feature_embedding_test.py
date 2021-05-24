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
import sys
import os
import numpy as np
import torch

# Dataset
from torch.utils.data import DataLoader
# stanford cloud segmentation
from datasets.S3DIS import *
# scannet cloud segmentation
from datasets.Scannet import *
# cloud segmentation using rgbd pcd from 7 scenes
from datasets.SinglePLY import *
# SLAM segmentation
from datasets.ScannetSLAM import *
# vlad
from datasets.ScannetTriple import *

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPFCNN
from models.PRNet import PRNet

# Visualisation
import open3d as o3d
import matplotlib.pyplot as plt


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

    print('\nEnvironment Setup')
    print('*****************')

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    # chosen_log = 'last_S3DIS'
    # chosen_log = 'results/Log_2021-04-19_10-01-23'  # => S3DIS, batch 4, 1st feat 64, 0.04-2.0
    # chosen_log = 'results/Log_2021-05-05_06-19-46'  # => ScanNet (subset), batch 10, 1st feat 64, 0.04-2.0
    chosen_log = 'results/Log_2021-05-14_02-21-27'  # => ScanNetSLAM (subset), batch 8, 1st feat 64, 0.04-2.0
    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    print('Chosen log:', chosen_log, chkp_idx)
    
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
    print('Checkpoints found:', chkps)
    # print(np.sort(chkps)) # sort string in alphbatic order

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)
    print('Checkpoints chosen:', chosen_chkp)

    ##################################
    # Change model parameters for test
    ##################################

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    # Change parameters for the test here. For example, you can stop augmenting the input data.
    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    config.batch_num = 1        # for cloud segmentation
    config.val_batch_num = 1    # for SLAM segmentation
    #config.in_radius = 4
    config.validation_size = 50    # decide how many points will be covered in prediction -> how many forward passes
    # 50 is a suitable value to cover a room-scale point cloud
    # 4 is a suitable value to cover a rgbd slam input size point cloud
    # set to number of frames in SLAM segmentaion
    config.input_threads = 0
    config.print_current()


    #################################################################################
    
    ##############
    # Prepare Data
    ##############

    print('\nData Preparation')
    print('****************')
    # new dataset for triplet input
    train_dataset = ScannetTripleDataset(config, 'training', balance_classes=False)
    # val_dataset = ScannetTripleDataset(config, 'validation', balance_classes=False)
    
    # train_batch = train_dataset[0]
    # val_batch = val_dataset[0]
    # # print(test_batch)

    # Initialize samplers
    train_sampler = ScannetTripleSampler(train_dataset)
    # val_sampler = ScannetTripleSampler(val_dataset)

    # Initialize the dataloader
    train_loader = DataLoader(train_dataset,
                                 batch_size=1,
                                 sampler=train_sampler,
                                 collate_fn=ScannetTripleCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    # val_loader = DataLoader(val_dataset,
    #                          batch_size=1,
    #                          sampler=val_sampler,
    #                          collate_fn=ScannetTripleCollate,
    #                          num_workers=config.input_threads,
    #                          pin_memory=True)

    # ## test
    # for i, batch in enumerate(train_loader):
    #     print(batch.lengths)
    #     break

    # Calibrate samplers
    train_sampler.calibration(train_loader, verbose=True)
    # val_sampler.calibration(val_loader, verbose=True)
    print('Calibed batch limit:', train_sampler.dataset.batch_limit)
    print('Calibed neighbor limit:', train_sampler.dataset.neighborhood_limits)
    


    print('\nModel Preparation')
    print('*****************')
    t1 = time.time()

    ##############
    # Architecture
    ##############

    if config.dataset_task in ['cloud_segmentation', 'slam_segmentation', 'registration']:
        seg_net = KPFCNN(config, train_dataset.label_values, train_dataset.ignored_labels)
        reg_net = PRNet(config)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)
    
    ############
    # Parameters
    ############

    # Choose to train on CPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    seg_net.to(device)

    reg_net.to(device)

    # for k, v in seg_net.named_parameters():
    #     print(k, v)    
    # print(seg_net.named_parameters())
    # for k, v in reg_net.named_parameters():
    #     print(k, v)
    # print(reg_net.named_parameters())

    optimizer = torch.optim.SGD(reg_net.parameters(),
                                lr=0.01,
                                momentum=0.98,
                                weight_decay=0.001)

    ##########################
    # Load previous checkpoint
    ##########################

    # load pretrained weights
    checkpoint = torch.load(chosen_chkp)
    # print(checkpoint.keys())    # ['epoch', 'model_state_dict', 'optimizer_state_dict', 'saving_path']
    # print(checkpoint['model_state_dict'].keys())    # where weights are stored
    # print(checkpoint['optimizer_state_dict'].keys())
    seg_net.load_state_dict(checkpoint['model_state_dict'])
    # number of epoch trained
    epoch = checkpoint['epoch']
    # set to evaluation mode
    seg_net.eval()
    # Dropout and BatchNorm (and maybe some custom modules) behave differently during training and 
    # evaluation. You must let the model know when to switch to eval mode by calling .eval() on 
    # the model. This sets self.training to False for every module in the model. 

    print("Model and training state restored with", epoch, "epoches trained.")
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    ####################
    # Train Next Network
    ####################

    saving_path = time.strftime('results/Recog_Log_%Y-%m-%d_%H-%M-%S', time.gmtime())

    # Checkpoints directory
    checkpoint_directory = join(saving_path, 'checkpoints')
    if not exists(checkpoint_directory):
        makedirs(checkpoint_directory)

    print('Initialize workers')
    t = time.time()
    epoch = 0
    break_cnt = 0
    for epoch in range(170):
        for i, batch in enumerate(train_loader):
            # new batch should be a stack of query batch, positive batches, negative batches, 7 pcds in total.

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

            if 'cuda' in device.type:
                batch.to(device)

            # # A Complete Forward Pass in SegNet
            # outputs = seg_net(batch, config)
            # Get interim features for place recognition
            # with dims [a, 64] [b, 128] [c, 256] [d, 512] [e, 1024]
            # inter_en_feat is a list of torch tensors
            inter_en_feat = seg_net.inter_encoder_features(batch, config)

            # separate the stacked features to different feature vectors
            # Dictionary of query, positive, negative point cloud features
            feat_vecs = {'query': [[]], 'positive': [[], []], 
                            'negative': [[], [], [], []]}
            feat_keys = ['query', 'positive', 'positive', 
                            'negative', 'negative', 'negative', 'negative']
            feats_idx = [0, 0, 1, 0, 1, 2, 3]
            # print('\n  ', len(inter_en_feat), 'intermediate features stored, size and length shown below:')
            for m, feat in enumerate(inter_en_feat):
                # feat = feat.to(torch.device("cpu"))
                layer_length = batch.lengths[m].to(torch.device("cpu"))
                # separate query, positive, and negative
                ind = 0
                # print(feat.size(), layer_length)
                for n, l in enumerate(layer_length):
                    one_feat = feat[ind:ind+l, :]
                    # store the feature vector in the corresponding place
                    key = feat_keys[n]
                    idx = feats_idx[n]
                    feat_vecs[key][idx].append(one_feat)
                    ind += l

            # zero the parameter gradients
            optimizer.zero_grad()

            # get vlad descriptor
            vlad_desp = []
            for key, vals in feat_vecs.items():
                for val in vals:
                    # print(key)
                    # for layer in range(5):
                    #     print(val[layer].size())
                    # print(val) # on cuda:0
                    descrip = reg_net(val)
                    # print('descriptor: ', descrip.size())
                    # print(descrip)
                    vlad_desp.append(descrip)

            # compute the loss
            loss = reg_net.loss(vlad_desp[0], vlad_desp[1:3], vlad_desp[3:])

            # backward + optimise
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)

            message = 'e{:03d}-i{:04d} => L={:.4f}'
            print(message.format(epoch, i, loss.item()))

            with open(join(config.saving_path, 'training.txt'), "a") as file:
                message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                file.write(message.format(epoch, i, loss, time.time() - t))

            # print('A complete forward step is done in {:.1f}s.'.format(time.time() - t))
            t = time.time()

            # ## Manually release gpu memory?
            # if 'cuda' in device.type:
            #     batch.to(torch.device("cpu"))
        #### END loop of batch
        print('Current epoch {:d} finished.'.format(epoch))


        # Get current state dict
        save_dict = {'epoch': epoch,
                     'model_state_dict': reg_net.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'saving_path': saving_path}
        # Save current state of the network (for restoring purposes)
        checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
        torch.save(save_dict, checkpoint_path)
        # Save checkpoints occasionally
        if (epoch + 1) % config.checkpoint_gap == 0:
            checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(epoch + 1))
            torch.save(save_dict, checkpoint_path)


    #################################################################################
    # ^ train recog net
    #################################################################################
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

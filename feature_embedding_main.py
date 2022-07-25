#
#
#      0=========================================0
#      |    Semantic Indoor Place Recognition    |
#      0=========================================0
#
#      Yuhang Ming
#

# Common libs
import enum
from multiprocessing import Value
import os
from re import search
# import sys
import signal
import argparse
import numpy as np
import torch

# Dataset
from torch.utils.data import DataLoader
from datasets.ScannetTriple import *

from models.architectures import KPFCNN
from models.PRNet import PRNet
from utils.config import Config
from utils.trainer import RecogModelTrainer

# VLAD test
from sklearn.neighbors import KDTree

# Visualisation
import open3d as o3d
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


if __name__ == '__main__':

    #####################
    # PARSE CMD-LINE ARGS
    #####################
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='bTRAIN', action='store_true', help='Set to train the VLAD layers')
    parser.add_argument('--test', dest='bTRAIN', action='store_false', help='Set to test the VLAD layers')
    parser.add_argument('--no_color', dest='bNoColor', action='store_true', help='Set not to use color in input point clouds')
    parser.add_argument('--optimiser', type=str, default='Adam', help='Choose the optimiser for training')
    parser.add_argument('--loss', type=str, default='lazy_quadruplet', help='Choose the loss function for training')
    parser.add_argument('--num_feat', type=int, default=5, help='How many block features to use [default: 5]')
    parser.add_argument('--evaluate', dest='bEVAL', action='store_true', help='Set to evaluate the VLAD results')
    parser.add_argument('--visualise', dest='bVISUAL', action='store_true', help='Set to visualise the VLAD results')
    FLAGS=parser.parse_args()
    if FLAGS.bTRAIN:
        print('Training parameters:')
        print('Optimiser:', FLAGS.optimiser)
        print('Number of features:', FLAGS.num_feat)
    else:
        print('Testing parameters load from files.')
        print('Evaluation:', FLAGS.bEVAL)
        print('Visualisation:', FLAGS.bVISUAL)

    ######################
    # LOAD THE PRE-TRAINED 
    # SEGMENTATION NETWORK
    ######################

    print('\nLoad pre-trained segmentation KP-FCNN')
    print('*************************************')
    t = time.time()
    if FLAGS.bNoColor:
        print('ScanNetSLAM, WITHOUT color')
        chosen_log = 'results/Log_2021-06-16_02-31-04'  # => ScanNetSLAM (full), w/o color, batch 8, 1st feat 64, 0.04-2.0
    else:
        print('ScanNetSLAM, WITH color')
        chosen_log = 'results/Log_2021-06-16_02-42-30'  # => ScanNetSLAM (full), with color, batch 8, 1st feat 64, 0.04-2.0
    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = 0 # chkp_500
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

    # Initialise and Load the segmentation network configs
    config = Config()
    config.load(chosen_log)
    config.KPlog = chosen_chkp
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
    # config.print_current()

    # set label manually here for scannet segmentation
    # with the purpose of putting loading parts together
    # ScanNet SLAM
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
    checkpoint = torch.load(config.KPlog)
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

    ###########################
    # TRAIN RECOGNITION NETWORK
    ###########################
    if FLAGS.bTRAIN:
        print('\nTRAINING VLAD Layer...\n')

        # update parameters for recog training
        config.num_feat = FLAGS.num_feat
        config.optimiser = FLAGS.optimiser
        config.loss = FLAGS.loss
        config.max_in_points = 9000
        config.max_val_points = 9000
        config.num_neg_samples = 6
        config.batch_num = 1
        config.val_batch_num = 1
        config.max_epoch = 30
        config.epoch_steps = 35000
        config.checkpoint_gap = 5
        # config.max_epoch = 175
        # config.epoch_steps = 5000
        # config.checkpoint_gap = 35        
        config.learning_rate = 1e-4
        config.lr_decays = {i: 0.9 for i in range(1, config.max_epoch)}

        config.weight_decay = 1e-3
        if config.saving:
            config.saving_path = time.strftime('results/Recog_Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        print('Updated max_in_p = ', config.max_in_points, config.max_val_points)
        print('        num neg samples =', config.num_neg_samples)
        print('        max_epoch / epoch_steps =', config.max_epoch, '/', config.epoch_steps)
        print('        checkpoint_gap =', config.checkpoint_gap)
        print('        batch_num train / val =', config.batch_num, '/', config.val_batch_num)
        print('        learning_rate =', config.learning_rate)
        print('        lr_decays =', config.lr_decays)
        print('        weight_decay =', config.weight_decay)
        print('        saving_path =', config.saving_path)

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
        # previous_training_path = 'Recog_Log_2021-08-20_22-39-43'
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


        print('\nPrepare Trainer')
        print('***************')
        # initialise trainier
        trainer = RecogModelTrainer(reg_net, config, chkp_path=chosen_chkp)
        
        print('\nStart training')
        print('**************')
        # TRAINING
        trainer.train(reg_net, seg_net, train_loader, config)
        print('Forcing exit now')
        os.kill(os.getpid(), signal.SIGINT)

    ##########################
    # TEST RECOGNITION NETWORK
    ##########################
    else:
        print('\nTESTING VLAD Layer...\n')

        print('\nLoad pre-trained recognition VLAD')
        print('*********************************')
        t = time.time()

        print('Quadruplet loss, feat_num = 5')
        # chosen_log = 'results/Recog_Log_2022-02-27_13-42-44'
        # chosen_log = 'results/Recog_Log_2021-08-29_13-46-24'
        chosen_log = 'results/Recog_Log_2021-07-29_17-53-02'

        # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
        chkp_idx = None        # USE current_ckpt, i.e. chkp_60
        print('Chosen log:', chosen_log, 'chkp_idx=', chkp_idx)

        # Find all checkpoints in the chosen training folder
        chkp_path = os.path.join(chosen_log, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']
        print('Checkpoints found:', np.sort(chkps))
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
        config.validation_size = 3700    # decide how many points will be covered in prediction -> how many forward passes
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
        # print('Test data:')
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


        db_path = join(chosen_log, 'database')
        if not exists(db_path):
            makedirs(db_path)
        
        dist_thred = 3.0
        # load database point clouds from file
        vlad_file = join(db_path, 'vlad_KDTree.txt')
        bIdfId_file = join(db_path, 'file_id.txt')
        bIdbId_file = join(db_path, 'batch_id.txt')
        if not exists(vlad_file):
            print('\nCreating database')
            print('*******************')
            t = time.time()
            # Get database
            break_cnt = 0
            database_vect = []
            batchInd_fileId = []
            batchInd_batchId = []
            database_cntr = {}
            db_count = 0
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

                ## NOTE centroid here is zero meaned. Use un-meaned pts for centroid test
                tmp_cntr = batch.frame_centers.cpu().detach().numpy()[0]    # np.array, (3,)
                tmp_fmid = batch.frame_inds.cpu().detach().numpy()[0]       # list, [scene index, frame index]
                tmp_pose = test_loader.dataset.poses[tmp_fmid[0]][tmp_fmid[1]]
                
                ## Load un-meaned pcd
                zmFile = test_loader.dataset.files[tmp_fmid[0]][tmp_fmid[1]].split('input_pcd_0mean')
                oriPCD = zmFile[0] + 'input_pcd' + zmFile[1]
                oriData = read_ply(oriPCD)
                oriPts = np.vstack((oriData['x'], oriData['y'], oriData['z'])).astype(np.float32).T # Nx3
                ori_cntr = np.mean(oriPts, axis=0)
                ori_cntr = tmp_pose[:3, :3] @ ori_cntr + tmp_pose[:3, 3]
                
                ## Use un-meaned centroid value
                tmp_cntr = ori_cntr
                # print(tmp_fmid, tmp_cntr, ori_cntr)
                if tmp_fmid[0] not in database_cntr.keys():
                    print('- ADDING NEW PCD TO DB:', tmp_fmid, db_count)
                    batch.to(device)
                    # get the VLAD descriptor
                    feat = seg_net.inter_encoder_features(batch)
                    vlad = reg_net(feat)
                    # store vlad vec, frm_cntr, and indices
                    database_vect.append(vlad.cpu().detach().numpy()[0]) # append a (1,256) np.ndarray
                    database_cntr[tmp_fmid[0]] = [tmp_cntr]
                    batchInd_fileId.append(tmp_fmid)
                    batchInd_batchId.append(i)

                    db_count += 1

                else:
                    # initialise boolean variable
                    bAddToDB = True

                    ## Only check with distance threshold
                    for db_cntr in database_cntr[tmp_fmid[0]]:
                        tmp_dist = np.linalg.norm(db_cntr - tmp_cntr)
                        if tmp_dist < dist_thred:
                            # skip if not enough movement detected
                            bAddToDB = False
                            break

                    if bAddToDB:
                        print('- ADDING NEW PCD TO DB:', tmp_fmid, db_count)
                        batch.to(device)
                        # get the VLAD descriptor
                        feat = seg_net.inter_encoder_features(batch)
                        vlad = reg_net(feat)
                        # store vlad vec, frm_cntr, and indices
                        database_vect.append(vlad.cpu().detach().numpy()[0]) # append a (1,256) np.ndarray
                        database_cntr[tmp_fmid[0]].append(tmp_cntr)
                        batchInd_fileId.append(tmp_fmid)
                        batchInd_batchId.append(i)

                        db_count += 1
           
                # print('stored center number:', len(database_cntr[tmp_fmid[0]]))
            database_vect = np.array(database_vect)

            print('DB size:', db_count, database_vect.shape)
            search_tree = KDTree(database_vect, leaf_size=4)
            # print(batchInd_fileId)
            # print(database_vect.shape)

            # store the database
            with open(vlad_file, "wb") as f:
                pickle.dump(search_tree, f)
            with open(bIdfId_file, "wb") as f:
                pickle.dump(batchInd_fileId, f)
            with open(bIdbId_file, "wb") as f:
                pickle.dump(batchInd_batchId, f)
            print('VLAD Databased SAVED to Files:', join(db_path, 'XXXX.txt'))

        else:
            # load the database
            # store the database
            with open(vlad_file, "rb") as f:
                search_tree = pickle.load(f)
            with open(bIdfId_file, "rb") as f:
                batchInd_fileId = pickle.load(f)
            with open(bIdbId_file, "rb") as f:
                batchInd_batchId = pickle.load(f)
            print('VLAD Databased LOADED from Files:', join(db_path, 'XXXX.txt'))
            db_vlad_vecs = np.array(search_tree.data, copy=False)
            print('Total stored submaps are:', db_vlad_vecs.shape)

        print('Done in {:.1f}s\n'.format(time.time() - t))

        # ## Uncomment here if needed
        # ## Visualise database point clouds
        # pre_sid = 0
        # fid_cnt = 0
        # # vis = o3d.visualization.Visualizer()
        # # vis.create_window(window_name='database', width=960, height=540, left=360, top=0)    
        # for sid, fid in batchInd_fileId:
        #     # get the query point cloud
        #     db_file = test_loader.dataset.files[sid][fid]
        #     db_file = db_file[:-4]+'_sub.ply'
        #     db_pose = test_loader.dataset.poses[sid][fid]
        #     print('processing:', db_file)
        #     print(db_pose)
        #     db_pcd = o3d.io.read_point_cloud(db_file)

        #     # vis = o3d.visualization.Visualizer()
        #     # vis.create_window(window_name='database', width=960, height=540, left=360, top=0)
        #     # db_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        #     # vis.add_geometry(db_pcd)
        #     # vis.run()
        #     # vis.destroy_window()

        #     db_pcd.transform(db_pose)
        #     trans = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
        #             [0.0, 0.0, 1.0, -1.*fid_cnt], [0.0, 0.0, 0.0, 1.0]]
        #     db_pcd.transform(trans)
        #     # visualise query in the original color
        #     # create visualisation window
        #     if sid != pre_sid:
        #         # hd_pcd_path = '/media/yohann/Datasets/datasets/ScanNet/scans'
        #         # pre_file = test_loader.dataset.files[pre_sid][fid]
        #         # scene = pre_file.split('/')[-2]
        #         # hd_pcd = o3d.io.read_point_cloud(join(hd_pcd_path, scene, scene + '_vh_clean.ply'))
        #         # vis.add_geometry(hd_pcd)
        #         trans = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
        #             [0.0, 0.0, 1.0, 1.*fid_cnt], [0.0, 0.0, 0.0, 1.0]]
        #         db_pcd.transform(trans)
        #         vis.run()
        #         vis.destroy_window()
        #         fid_cnt = 1
        #         pre_sid = sid
        #         vis = o3d.visualization.Visualizer()
        #         vis.create_window(window_name='database', width=960, height=540, left=360, top=0)
        #         vis.add_geometry(db_pcd)
        #     else:
        #         vis.add_geometry(db_pcd)
        #         fid_cnt += 1
        # vis.run()
        # vis.destroy_window()


        print('\nStart test')
        print('**********')
        t = time.time()
        # loop again to test with KDTree NN
        break_cnt = 0
        test_pair = []
        eval_results = []
        log_strings = ''
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

            # print('processing pcd no.', i)

            # skip if it's already stored in database
            # if i in batchInd_batchId or i%30 != 0:  # every 450 frame 
            if i in batchInd_batchId:
                print('Database frame, skipped.')
                continue
        
            tt = time.time()
            q_fmid = batch.frame_inds.cpu().detach().numpy()[0]     # list, [scene_id, frame_id]
            # Get VLAD descriptor
            batch.to(device)
            # print(' - Segmentation Layers')
            feat = seg_net.inter_encoder_features(batch)
            # print(' - VLAD Layers')
            vlad = reg_net(feat).cpu().detach().numpy()     # ndarray of (1, 256)
            # search for the closest match in DB
            dist, ind = search_tree.query(vlad, k=3)
            test_pair.append([q_fmid, 
                [batchInd_fileId[ ind[0][0] ],
                 batchInd_fileId[ ind[0][1] ],
                 batchInd_fileId[ ind[0][2] ]]
            ])

            if FLAGS.bEVAL:
                print('Evaluating...')
                q_cent = batch.frame_centers.cpu().detach().numpy()[0]    # np.array, (3,)
                # q_pcd_np = batch.points[0].cpu().detach().numpy()         # np.ndarray, (n, 3)
                q_pose = test_loader.dataset.poses[q_fmid[0]][q_fmid[1]]
                # q_pcd_np = (q_pose[:3, :3] @ q_pcd_np.T).T + q_pose[:3, 3]
                # queryKDT = KDTree(q_pcd_np)
                
                ## Use un-meaned pcd file
                query_file = test_loader.dataset.files[q_fmid[0]][q_fmid[1]].split('input_pcd_0mean')
                log_strings += (str(q_fmid[0]) + '_' + str(q_fmid[1]) + ': ' + query_file[1][1:] + '\n')
                oriPCD = query_file[0] + 'input_pcd' + query_file[1]
                oriData = read_ply(oriPCD)
                oriPts = np.vstack((oriData['x'], oriData['y'], oriData['z'])).astype(np.float32).T # Nx3
                ori_cntr = np.mean(oriPts, axis=0)
                ori_cntr = q_pose[:3, :3] @ ori_cntr + q_pose[:3, 3]
                q_cent = ori_cntr

                one_result = []
                for k, id in enumerate(ind[0]):
                    r_fmid = batchInd_fileId[id]
                    log_strings += ('--' + str(r_fmid[0]) + '_' + str(r_fmid[1]))
                    
                    # get k-th retrieved point cloud
                    retri_file = test_loader.dataset.files[r_fmid[0]][r_fmid[1]]
                    retri_file = retri_file.split('input_pcd_0mean')
                    r_pose = test_loader.dataset.poses[r_fmid[0]][r_fmid[1]]
                    
                    if r_fmid[0] != q_fmid[0]:
                        one_result.append(0)
                        log_strings += ': FAIL ' + retri_file[1][1:] + '\n'
                        continue
                    print(k, retri_file)

                    ## Use un-meaned pcd file
                    retriPCD = retri_file[0] + 'input_pcd' + retri_file[1]
                    retriData = read_ply(retriPCD)
                    retriPts = np.vstack((retriData['x'], retriData['y'], retriData['z'])).astype(np.float32).T # Nx3
                    r_cent = np.mean(retriPts, axis=0)
                    r_cent = r_pose[:3, :3] @ r_cent + r_pose[:3, 3]

                    # compute distance between centroids
                    dist = np.linalg.norm(q_cent - r_cent)
                    ## single threshold with only distance
                    if dist < dist_thred:
                        log_strings += ': SUCCESS ' + retri_file[1][1:] + ' ' + str(dist) + ' \n'
                        for fill in range(k, 3):
                            one_result.append(1)
                        break
                    else:
                        log_strings += ': FAIL ' + retri_file[1][1:] + ' ' + str(dist) + ' \n'
                        one_result.append(0)
                    
                eval_results.append(np.array(one_result))

            # print('current pcd finished in {:.4f}s'.format(time.time() - tt))
        print('Done in {:.1f}s\n'.format(time.time() - t))
        if FLAGS.bEVAL:
            eval_results = np.array(eval_results)
            num_test = eval_results.shape[0]
            accu_results = np.sum(eval_results, axis=0)
            print('Evaluation Results',
                  '\n    with', len(batchInd_fileId), 'stored pcd', num_test, 'test pcd',
                  '\n    with distance threshold', dist_thred)
            
            db_string = 'Database contains ' + str(len(batchInd_fileId)) + ' point clouds\n'
            qr_string = 'Total number of point cloud tested: ' + str(num_test) + '\n'
            thre_string = 'With distance threshold' + str(dist_thred) + '\n'
            result_strings = ''
            for k, accum1 in enumerate(accu_results):
                result_string = ' - Top ' + str(k+1) + ' recall = ' + str(accum1/num_test)
                print(result_string)
                result_strings += (result_string + '\n')
            
            # save logs to file
            text_file = open("detail_results.txt", "wt")
            text_file.write(log_strings)
            text_file.write('\n'+db_string)
            text_file.write(qr_string)
            text_file.write(thre_string)
            text_file.write(result_strings)
            text_file.close()

        if FLAGS.bVISUAL:
            print('\nVisualisation')
            print('*************')
            retri_colors = [[0, 0.651, 0.929],   # blue
                            [0, 0.8, 0],         # green
                            [1.0, 0.4, 0.4],     # red
                            [1, 0.706, 0]        # yellow
                            ]
            for query, retrivs in test_pair:
                # get the query point cloud
                query_file = test_loader.dataset.files[query[0]][query[1]]
                # q_pose = test_loader.dataset.poses[query[0]][query[1]]
                print('processing:', query_file)
                print('query/retrivs:', query, retrivs)

                q_pcd = o3d.io.read_point_cloud(query_file)
                # q_pcd.transform(q_pose)
                # q_pcd.paint_uniform_color([1, 0.706, 0])        # yellow
                q_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

                # visualise query in the original color
                # create visualisation window
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name='Query & Retrieval', width=960, height=960, left=360, top=0)
                vis.add_geometry(q_pcd)
                
                for k, retri in enumerate(retrivs):
                    # get k-th retrieved point cloud
                    retri_file = test_loader.dataset.files[retri[0]][retri[1]]
                    # r_pose = test_loader.dataset.poses[retri[0]][retri[1]]

                    r_pcd = o3d.io.read_point_cloud(retri_file)
                    trans = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 4.*(k+1)], [0.0, 0.0, 0.0, 1.0]]

                    # r_pcd.transform(r_pose)
                    r_pcd.transform(trans)
                    # r_pcd.paint_uniform_color(retri_colors[k])
                    r_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
                    vis.add_geometry(r_pcd)

                vis.run()
                vis.destroy_window()



    
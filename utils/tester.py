#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test of any model
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Yuhang Modification
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import enum
import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.metrics import confusion_matrix

#from utils.visualizer import show_ModelNet_models

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, chkp_path=None, on_gpu=True):

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        # load pretrained weights
        checkpoint = torch.load(chkp_path)
        # print(checkpoint.keys())    # ['epoch', 'model_state_dict', 'optimizer_state_dict', 'saving_path']
        # print(checkpoint['model_state_dict'].keys())    # where weights are stored
        # print(checkpoint['optimizer_state_dict'].keys())
        net.load_state_dict(checkpoint['model_state_dict'])
        # number of epoch trained
        self.epoch = checkpoint['epoch']
        # set to evaluation mode
        net.eval()
        # Dropout and BatchNorm (and maybe some custom modules) behave differently during training and 
        # evaluation. You must let the model know when to switch to eval mode by calling .eval() on 
        # the model. This sets self.training to False for every module in the model. 
        print("Model and training state restored.")

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def classification_test(self, net, test_loader, config, num_votes=100, debug=False):

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        self.test_probs = np.zeros((test_loader.dataset.num_models, nc_model))
        self.test_counts = np.zeros((test_loader.dataset.num_models, nc_model))

        t = [time.time()]
        mean_dt = np.zeros(1)
        last_display = time.time()
        while np.min(self.test_counts) < num_votes:

            # Run model on all test examples
            # ******************************

            # Initiate result containers
            probs = []
            targets = []
            obj_inds = []

            # Start validation loop
            for batch in test_loader:

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                # Get probs and labels
                probs += [softmax(outputs).cpu().detach().numpy()]
                targets += [batch.labels.cpu().numpy()]
                obj_inds += [batch.model_inds.cpu().numpy()]

                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Test vote {:.0f} : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(np.min(self.test_counts),
                                         100 * len(obj_inds) / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))
            # Stack all validation predictions
            probs = np.vstack(probs)
            targets = np.hstack(targets)
            obj_inds = np.hstack(obj_inds)

            if np.any(test_loader.dataset.input_labels[obj_inds] != targets):
                raise ValueError('wrong object indices')

            # Compute incremental average (predictions are always ordered)
            self.test_counts[obj_inds] += 1
            self.test_probs[obj_inds] += (probs - self.test_probs[obj_inds]) / (self.test_counts[obj_inds])

            # Save/Display temporary results
            # ******************************

            test_labels = np.array(test_loader.dataset.label_values)

            # Compute classification results
            C1 = fast_confusion(test_loader.dataset.input_labels,
                                np.argmax(self.test_probs, axis=1),
                                test_labels)

            ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
            print('Test Accuracy = {:.1f}%'.format(ACC))

        return

    def cloud_segmentation_test(self, net, test_loader, config, num_votes=100, debug=False):
        """
        Test method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        self.test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_labels]

        # Test saving path
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'predictions')):
                makedirs(join(test_path, 'predictions'))
            if not exists(join(test_path, 'probs')):
                makedirs(join(test_path, 'probs'))
            if not exists(join(test_path, 'potentials')):
                makedirs(join(test_path, 'potentials'))
        else:
            test_path = None

        # If on validation directly compute score
        if test_loader.dataset.set == 'validation':
            val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in test_loader.dataset.label_values:
                if label_value not in test_loader.dataset.ignored_labels:
                    val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                 for labels in test_loader.dataset.validation_labels])
                    i += 1
        else:
            val_proportions = None

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_min = -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                t += [time.time()]

                # Get probs and labels
                stacked_probs = softmax(outputs).cpu().detach().numpy()
                s_points = batch.points[0].cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                torch.cuda.synchronize(self.device)

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    points = s_points[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]

                    if 0 < test_radius_ratio < 1:
                        mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                        inds = inds[mask]
                        probs = probs[mask]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                    i0 += length

                # Average timing
                t += [time.time()]
                if i < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2])))

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.min_potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
            #print([np.mean(pots) for pots in test_loader.dataset.potentials])

            # Save predicted cloud
            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                if test_loader.dataset.set == 'validation':
                    print('\nConfusion on sub clouds')
                    Confs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Insert false columns for ignored labels
                        probs = np.array(self.test_probs[i], copy=True)
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                        # Targets
                        targets = test_loader.dataset.input_labels[i]

                        # Confs
                        Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print(s + '\n')

                # Save real IoU once in a while
                if int(np.ceil(new_min)) % 10 == 0:

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    proj_probs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        print(i, file_path, test_loader.dataset.test_proj[i].shape, self.test_probs[i].shape)

                        print(test_loader.dataset.test_proj[i].dtype, np.max(test_loader.dataset.test_proj[i]))
                        print(test_loader.dataset.test_proj[i][:5])

                        # Reproject probs on the evaluations points
                        probs = self.test_probs[i][test_loader.dataset.test_proj[i], :]
                        proj_probs += [probs]

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                    # Show vote results
                    if test_loader.dataset.set == 'validation':
                        print('Confusion on full clouds')
                        t1 = time.time()
                        Confs = []
                        for i, file_path in enumerate(test_loader.dataset.files):

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)

                            # Get the predicted labels
                            preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                            # Confusion
                            targets = test_loader.dataset.validation_labels[i]
                            Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                            if label_value in test_loader.dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        print('-' * len(s))
                        print(s)
                        print('-' * len(s) + '\n')

                    # Save predictions
                    print('Saving clouds')
                    t1 = time.time()
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Get file
                        points = test_loader.dataset.load_evaluation_points(file_path)

                        # Get the predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                        # Save plys
                        cloud_name = file_path.split('/')[-1]
                        test_name = join(test_path, 'predictions', cloud_name)
                        write_ply(test_name,
                                  [points, preds],
                                  ['x', 'y', 'z', 'preds'])
                        test_name2 = join(test_path, 'probs', cloud_name)
                        prob_names = ['_'.join(test_loader.dataset.label_to_names[label].split())
                                      for label in test_loader.dataset.label_values]
                        write_ply(test_name2,
                                  [points, proj_probs[i]],
                                  ['x', 'y', 'z'] + prob_names)

                        # Save potentials
                        pot_points = np.array(test_loader.dataset.pot_trees[i].data, copy=False)
                        pot_name = join(test_path, 'potentials', cloud_name)
                        pots = test_loader.dataset.potentials[i].numpy().astype(np.float32)
                        write_ply(pot_name,
                                  [pot_points.astype(np.float32), pots],
                                  ['x', 'y', 'z', 'pots'])

                        # Save ascii preds
                        if test_loader.dataset.set == 'test':
                            if test_loader.dataset.name.startswith('Semantic3D'):
                                ascii_name = join(test_path, 'predictions', test_loader.dataset.ascii_files[cloud_name])
                            else:
                                ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
                            np.savetxt(ascii_name, preds, fmt='%d')

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return

    def segmentation_with_return(self, net, test_loader, config, num_votes=100, debug=False):
        """
        Test method for cloud segmentation models with 
        prediction and intermediate features returned
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        self.test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_colors]
        print(len(self.test_probs))
        print(self.test_probs[0].shape) # (num of subsampled pts, num of classes - number of ignored class)
                                        # e.g. (58361, 13)

        # Test saving path
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'predictions')):
                makedirs(join(test_path, 'predictions'))
            if not exists(join(test_path, 'probs')):
                makedirs(join(test_path, 'probs'))
            if not exists(join(test_path, 'potentials')):
                makedirs(join(test_path, 'potentials'))
        else:
            test_path = None

        # If on validation directly compute score
        if test_loader.dataset.set == 'validation':
            val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in test_loader.dataset.label_values:
                if label_value not in test_loader.dataset.ignored_labels:
                    val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                 for labels in test_loader.dataset.validation_labels])
                    i += 1
        else:
            val_proportions = None

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_min = -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)
                # # all input points are centered at the center point. (normalised coordinates)
                # print('input points:', batch.points[0].size())
                # print('input features:', batch.features.size())
                # # center_inds is the index of the center point of each input regions (sub point cloud)
                # # in the coarser KDTree data
                # print('center index:', batch.center_inds.cpu().numpy()[0], cent_idx)
                # # print('input center: [{:.4f}, {:.4f}, {:.4f}], {:d}'.format(
                # #     batch.points[0][batch.center_inds[0]][0],
                # #     batch.points[0][batch.center_inds[0]][1],
                # #     batch.points[0][batch.center_inds[0]][2],
                # #     batch.center_inds[0]
                # # ))
                # print('output predictions:', outputs.size())
                
                # Get intermediate features
                # with dims [a, 64] [b, 128] [c, 256] [d, 512] [e, 1024]
                inter_en_feat = net.inter_encoder_features(batch, config)
                # print(len(inter_en_feat), 'intermediate features stored', end=': ')
                # for feat in inter_en_feat:
                #     # print(feat)
                #     print('[{:d}, {:d}]'.format(feat.size(0), feat.size(1)), end=' ')
                # print('')

                t += [time.time()]

                # Get probs and labels
                stacked_probs = softmax(outputs).cpu().detach().numpy() # predicted probability
                s_points = batch.points[0].cpu().numpy()                # subsampled point coordinates
                lengths = batch.lengths[0].cpu().numpy()                # num of pts for each point cloud
                in_inds = batch.input_inds.cpu().numpy()                # indices of each subsampled pts in the full subsampled point cloud
                cloud_inds = batch.cloud_inds.cpu().numpy()             # indices of input cloud
                torch.cuda.synchronize(self.device)
                cent_i = batch.center_inds.cpu().numpy()[0]
                cent_idx = np.where( in_inds == cent_i )
                print(cent_i, cent_idx)

                # Get predictions and labels per instance
                # ***************************************

                # print(lengths, '\n\n')    # in current test setting, only one value stored here, meaning one cloud only
                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    # All these predictions are in subsampled point cloud !!!!
                    points = s_points[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]

                    if 0 < test_radius_ratio < 1:
                        # axis=1, sum in each rows
                        # calculating the distance of each point to the center point
                        # take as inlier if dist < ratio*in_radius (0.7*2 in this case)
                        mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                        # print(mask)
                        # print(mask.shape[0], np.sum(mask))
                        inds = inds[mask]
                        probs = probs[mask]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                    i0 += length

                # Average timing
                t += [time.time()]
                if i < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                # if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})\n'
                print(message.format(test_epoch, i,
                                        100 * i / config.validation_size,
                                        1000 * (mean_dt[0]),
                                        1000 * (mean_dt[1]),
                                        1000 * (mean_dt[2])))

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.min_potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}, last_min+1 = {:.1f}'.format(test_epoch, new_min, last_min+1))
            #print([np.mean(pots) for pots in test_loader.dataset.potentials])

            # Update last_min
            last_min += 1

            # Show vote results (On subcloud so it is not the good values here)
            if test_loader.dataset.set == 'validation':
                print('\nConfusion on sub clouds')
                Confs = []
                for i, file_path in enumerate(test_loader.dataset.files):
                    # Insert false columns for ignored labels
                    probs = np.array(self.test_probs[i], copy=True)
                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            probs = np.insert(probs, l_ind, 0, axis=1)
                    # Predicted labels
                    preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                    # Targets
                    targets = test_loader.dataset.input_labels[i]
                    # Confs
                    Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]
                # Regroup confusions
                C = np.sum(np.stack(Confs), axis=0).astype(np.float32)
                # Remove ignored labels from confusions
                for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                    if label_value in test_loader.dataset.ignored_labels:
                        C = np.delete(C, l_ind, axis=0)
                        C = np.delete(C, l_ind, axis=1)
                # Rescale with the right number of point per class
                C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)
                # Compute IoUs
                IoUs = IoU_from_confusions(C)
                mIoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * mIoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                print(s + '\n')
            # print('new_win = {:.1f}, {:.1f}'.format(new_min, np.ceil(new_min)))

            # Project predictions
            print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
            t1 = time.time()
            proj_probs = []
            for i, file_path in enumerate(test_loader.dataset.files):
                # index, in case multiple point clouds are processed together
                # path to current point cloud's file
                # test_proj[i]: for each point in the original point cloud, the index of its corresponding subsampled point
                # number of subsampled points
                print(i, file_path, test_loader.dataset.test_proj[i].shape, self.test_probs[i].shape)
                # print(np.min(test_loader.dataset.test_proj[i])) # 0
                # print(np.max(test_loader.dataset.test_proj[i])) # 58360
                print(test_loader.dataset.test_proj[i].dtype, np.max(test_loader.dataset.test_proj[i]))
                print(test_loader.dataset.test_proj[i][:5]) # example output
                
                # Reproject probs on the evaluations points (the original point cloud)
                # now 'probs' should have rows as the number of orignal points
                # e.g. (82144, )
                probs = self.test_probs[i][test_loader.dataset.test_proj[i], :]

                # append the list
                proj_probs += [probs]
            t2 = time.time()
            print('Done in {:.4f} s\n'.format(t2 - t1))

            # Show vote results
            if test_loader.dataset.set == 'validation':
                print('Confusion on full clouds')
                t1 = time.time()
                Confs = []
                for i, file_path in enumerate(test_loader.dataset.files):
                    # Insert false columns for ignored labels
                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)
                    # Get the predicted labels
                    preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)
                    # Confusion
                    targets = test_loader.dataset.validation_labels[i]
                    Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]
                t2 = time.time()
                print('Done in {:.1f} s\n'.format(t2 - t1))
                # Regroup confusions
                C = np.sum(np.stack(Confs), axis=0)
                # Remove ignored labels from confusions
                for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                    if label_value in test_loader.dataset.ignored_labels:
                        C = np.delete(C, l_ind, axis=0)
                        C = np.delete(C, l_ind, axis=1)
                IoUs = IoU_from_confusions(C)
                mIoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * mIoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                print('-' * len(s))
                print(s)
                print('-' * len(s) + '\n')

            # Save predictions
            print('Saving clouds')
            t1 = time.time()
            print(test_loader.dataset.files)
            for i, file_path in enumerate(test_loader.dataset.files):
                cloud_name = file_path.split('/')[-1]
                
                # for scannet only:
                cloud = test_loader.dataset.clouds[i]
                # print(cloud, cloud_name)
                # print(type(file_path))
                file_path = join(test_loader.dataset.ply_path, test_loader.dataset.set+'_meshes', cloud + '_mesh.ply')
                # print(file_path)
                # print(type(file_path))

                # Get points from the file
                points = test_loader.dataset.load_evaluation_points(file_path)
                print(points.shape) # (821442, 3)
                # Get the predicted labels
                # For each row, find the index of the maximum probability
                # note remove ignored label before hand
                final_label = []
                for l in test_loader.dataset.label_values:
                    if l not in test_loader.dataset.ignored_labels:
                        final_label += [l]
                # for il in test_loader.dataset.ignored_labels:
                final_label = np.array(final_label)
                print(final_label)
                # print(final_label.shape)
                # print(test_loader.dataset.label_values.shape)

                # ignored labels causes problem, remove them
                # preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)
                preds = final_label[np.argmax(proj_probs[i], axis=1)].astype(np.int32)
                # print(test_loader.dataset.label_values) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12]
                # print(preds.shape)  # (821442,)
                
                # Save plys  
                test_name = join(test_path, 'predictions', cloud_name)
                print(points.shape)
                print(preds.shape)
                write_ply(test_name,
                            [points, preds],
                            ['x', 'y', 'z', 'preds'])
                test_name2 = join(test_path, 'probs', cloud_name)
                prob_names = ['_'.join(test_loader.dataset.label_to_names[label].split())
                                for label in test_loader.dataset.label_values]
                write_ply(test_name2,
                            [points, proj_probs[i]],
                            ['x', 'y', 'z'] + prob_names)
                test_name3 = join(test_path, 'predictions', cloud_name[:-4]+'_pred')
                # colour = np.array([np.array(object_label[i]).astype(np.uint8) for i in preds])
                colour = np.array(
                    [np.array(test_loader.dataset.label_to_colour[i]).astype(np.uint8) 
                    for i in preds])
                write_ply(test_name3,
                            [points, colour, preds],
                            ['x', 'y', 'z', 'red', 'green', 'blue', 'preds'])
                # Save potentials
                pot_points = np.array(test_loader.dataset.pot_trees[i].data, copy=False)
                pot_name = join(test_path, 'potentials', cloud_name)
                pots = test_loader.dataset.potentials[i].numpy().astype(np.float32)
                # print(pot_points.shape)     # (2631, 3) -> w.r.t. coarser_KDTree
                # print(pots.shape)           # (2631,)
                write_ply(pot_name,
                            [pot_points.astype(np.float32), pots],
                            ['x', 'y', 'z', 'pots'])
                # Save ascii preds
                if test_loader.dataset.set == 'test':
                    if test_loader.dataset.name.startswith('Semantic3D'):
                        ascii_name = join(test_path, 'predictions', test_loader.dataset.ascii_files[cloud_name])
                    else:
                        ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
                    np.savetxt(ascii_name, preds, fmt='%d')
                t2 = time.time()
                print('Done in {:.1f} s\n'.format(t2 - t1))

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break
            else:
                print(last_min, ' vs ', num_votes)

        return

    def slam_segmentation_test(self, net, test_loader, config, num_votes=100, debug=True):
        """
        Test method for slam segmentation models
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.5
        last_min = -0.5
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes
        nc_model = net.C

        # Test saving path
        test_path = None
        report_path = None
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])

            print("\n\ntest_path:", test_path)
            
            if not exists(test_path):
                makedirs(test_path)
            report_path = join(test_path, 'reports')
            if not exists(report_path):
                makedirs(report_path)

        if test_loader.dataset.set == 'validation':
            for folder in ['val_predictions', 'val_probs']:
                print("sub_folders:", join(test_path, folder))

                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))
        else:
            for folder in ['predictions', 'probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))

        # Init validation container
        all_f_preds = []
        all_f_labels = []
        if test_loader.dataset.set == 'validation' and test_loader.dataset.name == 'SemanticKitti':
            for i, seq_frames in enumerate(test_loader.dataset.frames):
                all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
                all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        test_epoch = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):

                # stop fetch new batch if no more points left
                if len(batch.points) == 0:
                    break

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                # # Get intermediate features
                # # with dims [a, 64] [b, 128] [c, 256] [d, 512] [e, 1024]
                # inter_en_feat = net.inter_encoder_features(batch, config)
                # print('    ', len(inter_en_feat), 'intermediate features stored', end=': ')
                # for feat in inter_en_feat:
                #     # print(feat)
                #     print('[{:d}, {:d}]'.format(feat.size(0), feat.size(1)), end=' ')
                # print('')

                # Get probs and labels
                stk_probs = softmax(outputs).cpu().detach().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                f_inds = batch.frame_inds.cpu().numpy()
                r_inds_list = batch.reproj_inds
                r_mask_list = batch.reproj_masks
                labels_list = batch.val_labels
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    probs = stk_probs[i0:i0 + length]
                    proj_inds = r_inds_list[b_i]
                    proj_mask = r_mask_list[b_i]
                    frame_labels = labels_list[b_i]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Project predictions on the frame points
                    # print(probs)
                    # print(proj_inds)
                    # print(proj_mask)
                    if test_loader.dataset.name == 'SemanticKitti':
                        proj_probs = probs[proj_inds]
                    elif test_loader.dataset.name == 'ScannetSLAM':
                        proj_probs = probs
                    else:
                        raise ValueError('Unknown dataset', test_loader.dataset.name)

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_probs = np.expand_dims(proj_probs, 0)

                    # Save probs in a binary file (uint8 format for lighter weight)
                    if test_loader.dataset.set == 'validation':
                        folder = 'val_probs'
                        pred_folder = 'val_predictions'
                    else:
                        folder = 'probs'
                        pred_folder = 'predictions'
                    
                    if test_loader.dataset.name == 'SemanticKitti':
                        seq_name = test_loader.dataset.scenes[s_ind]
                        filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                    elif test_loader.dataset.name == 'ScannetSLAM':
                        ori_file = test_loader.dataset.files[s_ind][f_ind].split('/')[-1]
                        filename = ori_file[:-4]+'.npy'
                    else:
                        raise ValueError('Unknown dataset', test_loader.dataset.name)
                    
                    filepath = join(test_path, folder, filename)
                    # print('filename', filename)
                    # print('filepath', filepath)
                    if exists(filepath):
                        frame_probs_uint8 = np.load(filepath)
                    else:
                        if test_loader.dataset.name == 'SemanticKitti':
                            frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                        elif test_loader.dataset.name == 'ScannetSLAM':
                            frame_probs_uint8 = np.zeros((proj_probs.shape[0], nc_model), dtype=np.uint8)
                        else:
                            raise ValueError('Unknown dataset', test_loader.dataset.name)
                    
                    if test_loader.dataset.name == 'SemanticKitti':
                        frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                        frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                        frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)
                    elif test_loader.dataset.name == 'ScannetSLAM':
                        frame_probs = frame_probs_uint8.astype(np.float32) / 255
                        frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                        frame_probs_uint8 = (frame_probs * 255).astype(np.uint8)
                    else:
                        raise ValueError('Unknown dataset', test_loader.dataset.name)
                    np.save(filepath, frame_probs_uint8)

                    # Save some prediction in ply format for visual
                    if test_loader.dataset.set == 'validation':

                        # Insert false columns for ignored labels
                        frame_probs_uint8_bis = frame_probs_uint8.copy()
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis,
                                                                                 axis=1)].astype(np.int32)

                        # Save some of the frame pots
                        if f_ind % 1 == 0:
                            if test_loader.dataset.name == 'SemanticKitti':
                                seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.scenes[s_ind])
                                velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                                frame_points = np.fromfile(velo_file, dtype=np.float32)
                                frame_points = frame_points.reshape((-1, 4))
                            elif test_loader.dataset.name == 'ScannetSLAM':
                                file_name = test_loader.dataset.files[s_ind][f_ind]
                                if not 'sub' in file_name:
                                    file_name = file_name[:-4]+'_sub.ply'
                                data = read_ply(file_name)
                                frame_points = np.vstack((data['x'], data['y'], data['z'])).T 
                            else:
                                raise ValueError('Unknown dataset', test_loader.dataset.name)

                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                print('saving with pots')
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds, pots],
                                          ['x', 'y', 'z', 'gt', 'pre', 'pots'])
                            else:
                                print('saving without pots')
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds],
                                          ['x', 'y', 'z', 'gt', 'pre'])

                            # Also Save lbl probabilities
                            probpath = join(test_path, folder, filename[:-4] + '_probs.ply')
                            lbl_names = [test_loader.dataset.label_to_names[l]
                                         for l in test_loader.dataset.label_values
                                         if l not in test_loader.dataset.ignored_labels]
                            write_ply(probpath,
                                      [frame_points[:, :3], frame_probs_uint8],
                                      ['x', 'y', 'z'] + lbl_names)

                        # # keep frame preds in memory
                        # all_f_preds[s_ind][f_ind] = frame_preds
                        # all_f_labels[s_ind][f_ind] = frame_labels

                    else:

                        # Save some of the frame preds
                        if f_inds[b_i, 1] % 1 == 0:

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    frame_probs_uint8 = np.insert(frame_probs_uint8, l_ind, 0, axis=1)

                            # Predicted labels
                            frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8,
                                                                                     axis=1)].astype(np.int32)

                            # Load points
                            if test_loader.dataset.name == 'SemanticKitti':
                                seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.scenes[s_ind])
                                velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                                frame_points = np.fromfile(velo_file, dtype=np.float32)
                                frame_points = frame_points.reshape((-1, 4))
                            elif test_loader.dataset.name == 'ScannetSLAM':
                                # get points from the batch
                                frame_points = batch.points[0].cpu().detach().numpy()[i0:i0 + length]
                                # file_name = test_loader.dataset.files[s_ind][f_ind]
                                # if not 'sub' in file_name:
                                #     file_name = file_name[:-4]+'_sub.ply'
                                # data = read_ply(file_name)
                                # frame_points = np.vstack((data['x'], data['y'], data['z'])).T
                            else:
                                raise ValueError('Unknown dataset', test_loader.dataset.name)

                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            # print('predpath', predpath)
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds, pots],
                                          ['x', 'y', 'z', 'pre', 'pots'])
                            else:
                                # print(frame_points.shape, frame_preds.shape)
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds],
                                          ['x', 'y', 'z', 'pre'])

                    # Stack all prediction for this epoch
                    i0 += length

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f}) / pots {:d} => {:.1f}%'
                    min_pot = int(torch.floor(torch.min(test_loader.dataset.potentials)))
                    pot_num = torch.sum(test_loader.dataset.potentials > min_pot + 0.5).type(torch.int32).item()
                    current_num = pot_num + (i + 1 - config.validation_size) * config.val_batch_num
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2]),
                                         min_pot,
                                         100.0 * current_num / len(test_loader.dataset.potentials)))


            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))

            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                if test_loader.dataset.set == 'validation' and test_loader.dataset.name == 'SemanticKitti' and last_min % 1 == 0:

                    #####################################
                    # Results on the whole validation set
                    #####################################

                    # Confusions for our subparts of validation set
                    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
                    for i, (preds, truth) in enumerate(zip(predictions, targets)):

                        # Confusions
                        Confs[i, :, :] = fast_confusion(truth, preds, test_loader.dataset.label_values).astype(np.int32)


                    # Show vote results
                    print('\nCompute confusion')

                    val_preds = []
                    val_labels = []
                    t1 = time.time()
                    for i, seq_frames in enumerate(test_loader.dataset.frames):
                        val_preds += [np.hstack(all_f_preds[i])]
                        val_labels += [np.hstack(all_f_labels[i])]
                    val_preds = np.hstack(val_preds)
                    val_labels = np.hstack(val_labels)
                    t2 = time.time()
                    C_tot = fast_confusion(val_labels, val_preds, test_loader.dataset.label_values)
                    t3 = time.time()
                    print(' Stacking time : {:.1f}s'.format(t2 - t1))
                    print('Confusion time : {:.1f}s'.format(t3 - t2))

                    s1 = '\n'
                    for cc in C_tot:
                        for c in cc:
                            s1 += '{:7.0f} '.format(c)
                        s1 += '\n'
                    if debug:
                        print(s1)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C_tot = np.delete(C_tot, l_ind, axis=0)
                            C_tot = np.delete(C_tot, l_ind, axis=1)

                    # Objects IoU
                    val_IoUs = IoU_from_confusions(C_tot)

                    # Compute IoUs
                    mIoU = np.mean(val_IoUs)
                    s2 = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in val_IoUs:
                        s2 += '{:5.2f} '.format(100 * IoU)
                    print(s2 + '\n')

                    # Save a report
                    report_file = join(report_path, 'report_{:04d}.txt'.format(int(np.floor(last_min))))
                    str = 'Report of the confusion and metrics\n'
                    str += '***********************************\n\n\n'
                    str += 'Confusion matrix:\n\n'
                    str += s1
                    str += '\nIoU values:\n\n'
                    str += s2
                    str += '\n\n'
                    with open(report_file, 'w') as f:
                        f.write(str)

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return

    def intermedian_features(self, net, test_loader, config):
        """
        Method to return only inter-blocks features for slam segmentation models
        """

        ## NOT CALLED !!!

        ############
        # Initialize
        ############

        # Test saving path
        test_path = None
        report_path = None
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            # print("\n\ntest_path:", test_path)
            
            if not exists(test_path):
                makedirs(test_path)
            report_path = join(test_path, 'reports')
            if not exists(report_path):
                makedirs(report_path)

        if test_loader.dataset.set == 'validation':
            for folder in ['val_predictions', 'val_probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))
        else:
            for folder in ['predictions', 'probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))

        #####################
        # Network predictions
        #####################

        print('Initialize workers')
        t = time.time()
        break_cnt = 0
        for i, batch in enumerate(test_loader):
            # new batch should be a list of query batch, positive batches, negative batches, 7 pcds in total.

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

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # print('   length of each pcd in each layer in the batch')
            # for length in batch.lengths:
            #     print(length)

            # # Forward pass
            # outputs = net(batch, config)

            # Get intermediate features
            # with dims [a, 64] [b, 128] [c, 256] [d, 512] [e, 1024]
            # inter_en_feat is a list of torch tensors
            inter_en_feat = net.inter_encoder_features(batch, config)

            
            # Dictionary of query, positive, negative point cloud features
            feat_vecs = {'query': [[]], 'positive': [[], []], 
                         'negative': [[], [], [], []]}
            feat_keys = ['query', 'positive', 'positive', 
                         'negative', 'negative', 'negative', 'negative']
            feats_idx = [0, 0, 1, 0, 1, 2, 3]
            print('  ', len(inter_en_feat), 'intermediate features stored')
            for m, feat in enumerate(inter_en_feat):
                feat = feat.to(torch.device("cpu"))
                layer_length = batch.lengths[m].to(torch.device("cpu"))
                # separate query, positive, and negative
                ind = 0
                print(feat.size(), layer_length)
                for n, l in enumerate(layer_length):
                    one_feat = feat[ind:ind+l, :]
                    # store the feature vector in the corresponding place
                    key = feat_keys[n]
                    idx = feats_idx[n]
                    feat_vecs[key][idx].append(one_feat)
                    ind += l
            
            # # test
            # for key, vals in feat_vecs.items():
            #     for val in vals:
            #         print(key)
            #         for layer in range(5):
            #             print(val[layer].size())

            
            print('   done in {:.1f}s.'.format(time.time() - t))
            t = time.time()

            ## Manually release gpu memory?
            if 'cuda' in self.device.type:
                batch.to(torch.device("cpu"))
        
        return 
        return inter_en_feat


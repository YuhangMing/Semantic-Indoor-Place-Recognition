#
#
#      0=================================0
#      |           VLAD Layers           |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Yuhang Ming
#

from multiprocessing import Value
from models.blocks import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PRNet(nn.Module):
    """
    Class defining the place recognition netowrk
    """

    def __init__(self, config):
        super(PRNet, self).__init__()

        # Feature concatenation layer
        feature_size = config.first_features_dim *2 *2 *2 *2   # D
        in_size = config.first_features_dim
        self.num_feat = config.num_feat
        self.num_neg_samples = config.num_neg_samples
        # print(feature_size)
        # print(in_size)

        # FC layer stretching feat vector to the same dim
        # (N, *, H_in) -> (N, *, H_out)
        if self.num_feat == 5:
            self.FC_1 = UnaryBlock(in_size, feature_size, False, 0)
            self.FC_2 = UnaryBlock(in_size*2, feature_size, False, 0)
            self.FC_3 = UnaryBlock(in_size*(2**2), feature_size, False, 0)
            self.FC_4 = UnaryBlock(in_size*(2**3), feature_size, False, 0)
        elif self.num_feat == 3:
            self.FC_3 = UnaryBlock(in_size*(2**2), feature_size, False, 0)
            self.FC_4 = UnaryBlock(in_size*(2**3), feature_size, False, 0)
        elif self.num_feat == 1:
            pass
        else: 
            raise ValueError('unsupport feature number')

        # NetVLAD layer
        cluster_size = 64  # K
        output_dim = 256    
        max_num_pts = 25000
        self.vlad_layer = NetVLAD(feature_size=feature_size, max_samples=max_num_pts, 
                                  cluster_size=cluster_size, output_dim=output_dim, 
                                  gating=False, add_batch_norm=False)

        # # Network Losses
        # #### Triplet Loss
        # # Note: only difference between Torch built-in triplet loss and TF version PNV author's triplet loss
        # # is Torch use mean for reduction by default, while TF uses sum
        # # change Torch to use sum reduction by adding reduction='sum' 
        # self.criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2.0)


    def forward(self, feat_vec):
        # intpu a list of feature vectors 
        # (from each conv blocks)
        # print(feat_vec[0].size())
        # x_1 = self.FC_1(torch.unsqueeze(feat_vec[0], 0))    # single batch test, add back batch dimension
        
        # print('Input feature size per layer:', feat_vec[0].size(), feat_vec[1].size(), feat_vec[2].size(), 
        #                                  feat_vec[3].size(), feat_vec[4].size())
        # concatenate feature vectors from each conv block
        
        x_5 = feat_vec[4]
        if self.num_feat == 5:
            # print('using all 5 block features')
            x_1 = self.FC_1(feat_vec[0])
            x_2 = self.FC_2(feat_vec[1])
            x_3 = self.FC_3(feat_vec[2])
            x_4 = self.FC_4(feat_vec[3])
            # (N1+N2+N3+N4+N5 = N, 1024) [1, 11667, 1024]
            x = torch.cat((x_1, x_2, x_3, x_4, x_5), 0)
        elif self.num_feat == 3:
            x_3 = self.FC_3(feat_vec[2])
            x_4 = self.FC_4(feat_vec[3])
            # print('using last 3 block features')
            x = torch.cat((x_3, x_4, x_5), 0)
        elif self.num_feat == 1:
            x = x_5
        else:
            raise ValueError('unsupport feature number')
        
        # print('feature size per layer:', x_1.size(), x_2.size(), x_3.size(), x_4.size(), x_5.size())
        # print('cated feature size:', x.size())

        # get global descriptor
        x = self.vlad_layer(x)
        # print('vlad vec size:', x.size())

        return x
    
    def loss(self, loss_function, a, p, n, n_star=None):
        if loss_function == 'LzQuad':
            loss = self.LazyQuadrupletLoss(a, p, n, n_star)
        elif loss_function == 'LzTrip':
            loss, _ = self.TripletLoss(a, p, n, lazy=True)
        elif loss_function == 'Trip':
            loss, _ = self.TripletLoss(a, p, n)
        else:
            raise ValueError(('Unknown loss function', loss_function))
        
        return loss

        # # a: single tensor
        # # p: list of 2 tensors
        # # n: list of 18 tensors
        # # n_star: single tensor
        # # cat to meet tripletloss input requirement
        # n_neg_samples = self.num_neg_samples
        # anc = torch.cat(2*n_neg_samples*[a], dim=0)
        # # [pos0, ..., pos0, pos1, ..., pos1]
        # pos0 = torch.cat(n_neg_samples*[p[0]], dim=0)
        # pos1 = torch.cat(n_neg_samples*[p[1]], dim=0)
        # pos = torch.cat( (pos0, pos1), dim=0 )
        # # [neg0, ... neg8, neg0, ..., neg8]
        # neg = n[0]
        # for nIdx in range(1, n_neg_samples):
        #     neg = torch.cat( (neg, n[nIdx]), dim=0)
        # neg = torch.cat( (neg, neg), dim=0 )
        # if not n_star is None:
        #     neg_star = torch.cat(2*n_neg_samples*[n_star], dim=0)
        # # Triplet Loss
        # loss = self.criterion(anc, pos, neg)
        # return loss

    def LazyQuadrupletLoss(self, anc, pos, neg, neg_star, p=2.0, alpha=0.5, beta=0.2):
        # implementation of lazy quadruplet loss 
        # proposed in the PointNetVLAD
        
        # first is just a lazy triplet loss
        triplet_loss, best_pos = self.TripletLoss(anc, pos, neg, p=p, margin=alpha, lazy=True)

        # compute the second loss
        # compute p_distance
        neg_star = torch.cat(self.num_neg_samples * [neg_star], dim=0)
        neg = torch.cat(neg, dim=0)
        # delta_neg = self.p_distance(anc, neg, p=p)
        delta_neg = self.p_distance(neg_star, neg, p=p)
        # margined difference
        diff = best_pos + beta - delta_neg
        # zeros and set to device
        zeros = torch.zeros(delta_neg.size())
        bCUDA = delta_neg.get_device()
        if bCUDA >= 0 and torch.cuda.is_available():
            zeros = zeros.to(torch.device(bCUDA))
        hinge_loss = torch.max(zeros, diff)
        # LAZY
        second_loss = torch.max(hinge_loss)

        # get combined loss
        loss = triplet_loss + second_loss
        return loss

        # if anc.size() != pos.size() or anc.size() != neg.size() or anc.size() != neg_star.size():
        #     raise ValueError('Size of input tensors should match!!', anc.size(), pos.size(), neg.size(), neg_star.size())
        # # get distance values
        # delta_pos = torch.cdist(anc, pos, p=p)
        # delta_neg = torch.cdist(pos, neg, p=p)
        # delta_neg_star = torch.cdist(neg_star, neg, p=p)
        # zeros = torch.zeros(delta_pos.size())
        # bCUDA = delta_pos.get_device()
        # if bCUDA >= 0 and torch.cuda.is_available():
        #     zeros = zeros.to(torch.device(bCUDA))
        # # get the loss value
        # first_term = torch.max( torch.max(zeros, delta_pos + alpha - delta_neg) )
        # second_term = torch.max( torch.max(zeros, delta_pos + beta - delta_neg_star) )
        # loss = first_term + second_term
        # return loss

    def TripletLoss(self, anc, pos, neg, p=2.0, margin=0.5, lazy=False):
        # anc: anchor tensor
        # pos: list of positive tensors (x2)
        # neg: list of negative tensors (x6 or x8)
        # implementation of (Lazy)TripletLoss

        # Find best positive tensor (Focusing on best pos)
        anc = torch.cat([anc, anc], dim=0)
        pos = torch.cat([pos[0], pos[1]], dim=0)
        # print(anc.size(), pos.size())
        delta_pos = self.p_distance(anc, pos, p=p)
        # print('delta_pos', delta_pos.size(), '\n', delta_pos)
        best_pos_val = torch.min(delta_pos)
        # print('best_pos', best_pos_val)

        # Compute the hinge loss
        # doesn't need to be multiple of 2, stack to the same size is fine
        if self.num_neg_samples % 2 != 0:
            raise ValueError('num of negative samples should be integal multiplication of 2.')
        anc = torch.cat(int(self.num_neg_samples/2) * [anc], dim=0)
        neg = torch.cat(neg, dim=0)
        # print('\n', anc.size(), neg.size())
        delta_neg = self.p_distance(anc, neg, p=p)
        # print(delta_neg)
        # delta_neg = torch.cdist(anc, neg, p=p)
        # margined difference
        diff = best_pos_val + margin - delta_neg
        # zeros and set to device
        zeros = torch.zeros(delta_neg.size())
        bCUDA = delta_neg.get_device()
        if bCUDA >= 0 and torch.cuda.is_available():
            zeros = zeros.to(torch.device(bCUDA))
        hinge_loss = torch.max(zeros, diff)
        # print('hinge test:', hinge_loss, '\n', diff)

        # get final (lazy) triplet loss
        if lazy:
            loss = torch.max(hinge_loss)
        else:
            loss = torch.sum(hinge_loss)
        
        return loss, best_pos_val
        
    def p_distance(self, x, y, p=2.0):
        diff = x - y
        # p=2:
        if p == 1:
            dist = torch.abs(diff)
        elif p == 2:
            dist = torch.sqrt(torch.sum(diff**2, dim=1))
        else:
            raise ValueError('unsupport distance function p =',p, 'only p= 1 or 2 is supported')

        return dist


    # def accuracy(outputs):


# modified from https://github.com/cattaneod/PointNetVlad-Pytorch/blob/master/models/PointNetVlad.py
class NetVLAD(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        # self.max_samples = max_samples
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        
        # 1024 -> 64
        self.FC_1 = nn.Linear(feature_size, cluster_size, bias=True)
        self.cluster_centers = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        # 1024*64 -> 256
        self.FC_2 = nn.Linear(feature_size*cluster_size, output_dim, bias=True)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        # print('- dims of parameters:')
        # print('  input x:', x.size())                              # [730, 1024]   [num_pts, feat_dim]

        # print('- step 1: compute softmax assignment')
        ## w^T x + b
        alpha = self.FC_1(x)
        # print('  alpha:', alpha.size())
        alpha = self.softmax(alpha)
        # print('  softmax:', alpha.size())

        # Sum over all points
        # print('- step 2: compute vlad')
        first_term = torch.matmul(torch.transpose(x, 0, 1), alpha)
        # print('  first term:', first_term.size())
        second_term = torch.sum(alpha, 0) * self.cluster_centers
        # print('  second term:', second_term.size())
        vlad = first_term - second_term
        # intra-normalisation (column-wise L2 norm)
        vlad = F.normalize(vlad, dim=0, p=2)
        # reshape to (1x(DxK))
        vlad = vlad.view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)
        # print('  reshaped vlad vec:', vlad.size())

        # print('- step 2: dimension reduction')
        vlad = self.FC_2(vlad)
        # print('  vlad size:', vlad.size())
        vlad = F.normalize(vlad, dim=1, p=2)
        # print('  vlad size:', vlad.size())

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad

class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation














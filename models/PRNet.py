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

        # NetVLAD layer
        cluster_size = 64  # K
        output_dim = 256    
        max_num_pts = 25000
        self.vlad_layer = NetVLAD(feature_size=feature_size, max_samples=max_num_pts, 
                                  cluster_size=cluster_size, output_dim=output_dim, 
                                  gating=False, add_batch_norm=False)

        # Network Losses
        # Note: only difference between Torch built-in triplet loss and TF version PNV author's triplet loss
        # is Torch use mean for reduction by default, while TF uses sum
        # change Torch to use sum reduction by adding reduction='sum' 
        self.criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2.0)


    def forward(self, feat_vec):
        # intpu a list of feature vectors 
        # (from each conv blocks)
        # print(feat_vec[0].size())
        # x_1 = self.FC_1(torch.unsqueeze(feat_vec[0], 0))    # single batch test, add back batch dimension
        
        # print('Input feature size per layer:', feat_vec[0].size(), feat_vec[1].size(), feat_vec[2].size(), 
        #                                  feat_vec[3].size(), feat_vec[4].size())
        # concatenate feature vectors from each conv block
        x_3 = self.FC_3(feat_vec[2])
        x_4 = self.FC_4(feat_vec[3])
        x_5 = feat_vec[4]
        if self.num_feat == 5:
            print('using all 5 block features')
            x_1 = self.FC_1(feat_vec[0])
            x_2 = self.FC_2(feat_vec[1])
            # (N1+N2+N3+N4+N5 = N, 1024) [1, 11667, 1024]
            x = torch.cat((x_1, x_2, x_3, x_4, x_5), 0)
        elif self.num_feat == 3:
            print('using last 3 block features')
            x = torch.cat((x_3, x_4, x_5), 0)
        else:
            raise ValueError('unsupport feature number')
        
        # print('feature size per layer:', x_1.size(), x_2.size(), x_3.size(), x_4.size(), x_5.size())
        # print('cated feature size:', x.size())

        # get global descriptor
        x = self.vlad_layer(x)
        # print('vlad vec size:', x.size())

        return x
    
    def loss(self, a, p, n):
        # a: single tensor
        # p: list of 2 tensors
        # n: list of 18 tensors
        
        # First try: TripletLoss
        # cat to meet tripletloss input requirement
        n_neg_samples = self.num_neg_samples
        anc = torch.cat(2*n_neg_samples*[a], dim=0)
        pos0 = torch.cat(n_neg_samples*[p[0]], dim=0)
        pos1 = torch.cat(n_neg_samples*[p[1]], dim=0)
        pos = torch.cat( (pos0, pos1), dim=0 )
        # pos = torch.cat((p[0],p[0],p[0],p[0],
        #                  p[1],p[1],p[1],p[1]), dim=0)
        neg = n[0]
        for nIdx in range(1, n_neg_samples):
            neg = torch.cat( (neg, n[nIdx]), dim=0)
        neg = torch.cat( (neg, neg), dim=0 )
        # neg = torch.cat((n[0],n[1],n[2],n[3],
        #                  n[0],n[1],n[2],n[3]), dim=0)
        # compute loss
        loss = self.criterion(anc, pos, neg)
        
        return loss
    # def accuracy(outputs):


# modified from https://github.com/cattaneod/PointNetVlad-Pytorch/blob/master/models/PointNetVlad.py
class NetVLAD(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
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

        # # (1024, 128)
        # self.cluster_weights = nn.Parameter(torch.randn(
        #     feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        # # (1, 1024, 128)
        # self.cluster_weights2 = nn.Parameter(torch.randn(
        #     1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        # # (1024*128, 256)
        # self.hidden1_weights = nn.Parameter(
        #     torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        # if add_batch_norm:
        #     self.cluster_biases = None
        #     self.bn1 = nn.BatchNorm1d(cluster_size)
        # else:
        #     self.cluster_biases = nn.Parameter(torch.randn(
        #         cluster_size) * 1 / math.sqrt(feature_size))
        #     self.bn1 = None

        # WHY BATCH NORM?? not mentioned in the paper
        # self.bn2 = nn.BatchNorm1d(output_dim)

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

# gate what?
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














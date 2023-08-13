# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000, weight=None): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold] 
        return pixel_losses.mean()

# 改进的Focalloss函数
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, balance_param=0.25,ignore_label=-1, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_label = ignore_label
        self.balance_param=balance_param
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                input=score, size=(h, w), mode='bilinear')

        # cross_entropy = F.cross_entropy(score, target,weight=self.weight)
        # cross_entropy_log = torch.log(cross_entropy)

        # logpt = - F.cross_entropy(score, target)
        logpt = - self.criterion(score, target)
        pt = torch.exp(logpt)
        # compute the loss
        # loss = ((1 - pt) ** self.gamma) * logpt

        # averaging (or not) loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        # balanced_focal_loss = self.balance_param * focal_loss
        return focal_loss



class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, weight=None,ignore_label=-1, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.ignore_label = ignore_label
        self.size_average = size_average

    def forward(self, score, target):

        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                input=score, size=(h, w), mode='bilinear')

        # if score.dim() > 2:
        #     score = score.contiguous().view(score.size(0), score.size(1), -1)
        #     score = score.transpose(1, 2)
        #     score = score.contiguous().view(-1, score.size(2)).squeeze()
        # if target.dim() == 4:
        #     target = target.contiguous().view(target.size(0), target.size(1), -1)
        #     target = target.transpose(1, 2)
        #     target = target.contiguous().view(-1, target.size(2)).squeeze()
        # elif target.dim() == 3:
        #     target = target.view(-1)
        # else:
        #     target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        # 带权重的交叉熵损失函数
        logpt = -F.cross_entropy(score, target, weight=weight,ignore_index=self.ignore_label)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt) ** self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
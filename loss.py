import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from util import accuracy


class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s):
        """
        使用AAM-Softmax算法。
        参数：
        n_class -- 分类的数量
        m -- 调整类别之间的夹角
        s -- 圆的半径
        """
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        # Initialize the classification weight matrix
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        # Weight Initialization to one
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce = nn.CrossEntropyLoss()
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        # Calculate Cosine Similarity
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # Calculate the cosine distance of the positive and negative samples
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        # Cross entropy is used to calculate the loss
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), turek=(1,))[0]
        return loss, prec1


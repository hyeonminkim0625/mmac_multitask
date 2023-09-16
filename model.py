import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import timm
import numpy as np
from timm.layers import NormMlpClassifierHead,trunc_normal_, AvgPool2dSame, DropPath, Mlp, GlobalResponseNormMlp,LayerNorm2d, LayerNorm


class ConvModule(nn.Module):
    """Some Information about ConvModule"""
    def __init__(self,c1, c2, k, s=1, p=0, d=1, g=1):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, d, g, bias=False)
        self.norm = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(False)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks
    https://arxiv.org/abs/1901.02446
    """
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.lateral_convs = nn.ModuleList([])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[::-1]:
            self.lateral_convs.append(ConvModule(ch, channel, 1))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

    def forward(self, features):
        
        out = self.lateral_convs[0](features[0])
        
        for i in range(1, len(features)):
            out = F.interpolate(out, scale_factor=2.0, mode='bilinear')
            out = out + self.lateral_convs[i](features[i])
            out = self.output_convs[i](out)
            
        return out

class Model(nn.Module):
    def __init__(self,arch):
        super(Model, self).__init__()
        self.model = timm.create_model(arch, pretrained=True, num_classes=0, drop_path_rate=0.2,global_pool='',features_only=True)
        
        self.task2_fpn = FPNHead([96,192,384],128,128)

        self.task2_1 = nn.Conv2d(128, 2,kernel_size=1)
        self.task2_2 = nn.Conv2d(128, 2,kernel_size=1)
        self.task2_3 = nn.Conv2d(128, 2,kernel_size=1)

        self.task3 = NormMlpClassifierHead(
                768,
                1,
                hidden_size=None,
                pool_type='avg',
                drop_rate=0.0,
                norm_layer='layernorm2d',
                act_layer='gelu')

        self.task3_fc = self.task3.fc
        self.task3.fc = torch.nn.Identity()

        self.task1 = NormMlpClassifierHead(
                768,
                5,
                hidden_size=None,
                pool_type='avg',
                drop_rate=0.1,
                norm_layer='layernorm2d',
                act_layer='gelu')

        self.task1_fc = self.task1.fc
        self.task1.fc = torch.nn.Identity()
    
    def forward(self, imgs,task,lam):
        x1,x2,x3,x4 = self.model(imgs)

        """
        x1_new = x1.clone()
        x2_new = x2.clone()
        x3_new = x3.clone()
        
        x1_new[task==21] = x1[task==21] * lam + x1[task==21].flip(0) * (1. - lam)
        x1_new[task==22] = x1[task==22] * lam + x1[task==22].flip(0) * (1. - lam)
        x1_new[task==23] = x1[task==23] * lam + x1[task==23].flip(0) * (1. - lam)

        x2_new[task==21] = x2[task==21] * lam + x2[task==21].flip(0) * (1. - lam)
        x2_new[task==22] = x2[task==22] * lam + x2[task==22].flip(0) * (1. - lam)
        x2_new[task==23] = x2[task==23] * lam + x2[task==23].flip(0) * (1. - lam)

        x3_new[task==21] = x3[task==21] * lam + x3[task==21].flip(0) * (1. - lam)
        x3_new[task==22] = x3[task==22] * lam + x3[task==22].flip(0) * (1. - lam)
        x3_new[task==23] = x3[task==23] * lam + x3[task==23].flip(0) * (1. - lam)
        """
        x1 = self.task2_fpn([x3,x2,x1])

        x_task1_feature = self.task1(x4)
        x_task3_feature = self.task3(x4)
    
        x_task1_feature[task==1] = x_task1_feature[task==1] * lam + x_task1_feature[task==1].flip(0) * (1. - lam)
        x_task3_feature[task==3] = x_task3_feature[task==3] * lam + x_task3_feature[task==3].flip(0) * (1. - lam)

        x_task1 = self.task1_fc(x_task1_feature)
        x_task3 = self.task3_fc(x_task3_feature)

        x_task21 = self.task2_1(x1)
        x_task21 = F.interpolate(x_task21, size = (800,800), mode='bilinear')

        x_task22 = self.task2_2(x1)
        x_task22 = F.interpolate(x_task22, size = (800,800), mode='bilinear')

        x_task23 = self.task2_3(x1)
        x_task23 = F.interpolate(x_task23, size = (800,800), mode='bilinear')

        return x_task1,x_task21,x_task22,x_task23,x_task3,F.normalize(x_task1_feature,dim=1)
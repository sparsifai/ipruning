from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time,os,sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
import hsummary
import torchvision
from numpy import unique
from scipy.stats import entropy as scipy_entropy

def networksize():
    network_size = {'Conv2d-1':[64],'Conv2d-2':[192],'Conv2d-3':[384],'Conv2d-4':[256],'Conv2d-5':[256]} # for each layer: [number of featuremaps, size of inupt, size of output]
    return network_size

    
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)



class NetOrg(nn.Module):
    def __init__(self,s_input_channel,n_classes):
        super(NetOrg, self).__init__()
        self.pre_layers = nn.Sequential(
                    nn.Conv2d(s_input_channel, 192, kernel_size=3, padding=1),
                    nn.BatchNorm2d(192),
                    nn.ReLU(True),
                )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(2458624, n_classes) #1024

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)        
        out = self.linear(out)
        output = F.log_softmax(out, dim=1)

        return output

class Net(nn.Module):
    def __init__(self,s_input_channel,n_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(s_input_channel, 6, kernel_size=5) # (1, 32, kernel=3, 1)   # Tensor size: #channels,#FMaps,#ConvImageSize(e.g. 28-3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) 

        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # (9216, 128)
        self.fc2 = nn.Linear(120, 84) # (128, 10)
        self.fc3 = nn.Linear(84, n_classes) # (128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = torch.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = torch.flatten(x, 1)        
        # x = self.dropout1(x)
        x = self.fc1(x)
        x = torch.relu(x)
        # x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)
        logits = torch.relu(x)

        return output, sig1_x, sig2_x,logits


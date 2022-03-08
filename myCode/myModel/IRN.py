import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..") 
from utils import random_operation


"""
The residual Block of the image restruction network
    In the paper's codes,batchNorm uses decay.But pytorch's BN uses momentum
"""
class ResidualBlock(nn.Module):
    def __init__(self,in_planes,planes=64,stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes,eps=0.001)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes,eps=0.001)

    def forword(self,input):
        x = self.bn1(self.conv1(input))
        x = self.prelu(x)
        x = self.bn2(self.conv2(x))
        x = x + input
        return x


"""
The whole generator network
"""
class IRN(nn.Module):
    def __init__(self, in_planes=3, num_blocks=16):
        super(IRN, self).__init__()
        self.net1 = nn.Sequential(
                nn.Conv2d(in_planes, out_channels=64,
                          kernel_size=9, stride=1, bias=False),
                nn.PReLU()
            )
        # There are 16 residual blocks
        blocks = [ResidualBlock(in_planes=64)]
        for _ in range(1,num_blocks):
            blocks.append(ResidualBlock(in_planes=64))
        self.net2 = nn.Sequential(*blocks)
        self.net3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64,
                          kernel_size=3, stride=1, bias=False),
                nn.BatchNorm2d(num_features=64,eps=0.001)
            )
        self.net4 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256,
                          kernel_size=3, stride=1, bias=False),
                nn.PReLU(),
                nn.Conv2d(in_channels=256, out_channels=256,
                          kernel_size=3, stride=1, bias=False),
                nn.PReLU(),
                nn.Conv2d(in_channels=256, out_channels=3,
                          kernel_size=9, stride=1, bias=False)
            )
        
    def forword(self,input):
        x = self.net1(input)
        output1 = x
        x = self.net3(self.net2(x))
        x = x + output1
        x = self.net4(x)
        x = random_operation.random_resize(x)  # random resize
        #x = random_operation.random_pad(x)  # random pad
        return x
        




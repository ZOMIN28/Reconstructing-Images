import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms


# 随机改变大小，范围为[299,331]
def random_resize(input):
    num = int(torch.linspace((299,331),steps=1))
    trans = torchvision.transforms.RandomSizedCrop(num)
    return trans(input)


# 随机填充
# def random_pad(input):

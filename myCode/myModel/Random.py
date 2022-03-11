import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Randomization(nn.Module):
    '''
    Random Resizing: after pre-processing, resize the original image (size of 299 x 299 x 3) to a larger size,
    Rnd x Rnd x 3, randomly, where Rnd is within the range [310, 331).

    Random Padding: after resizing, pad the resized image to a new image with size 331 x 331 x 3, where the padding
    size at left, right, upper, bottom are [a, 331-Rnd-a, b, 331-Rnd-b]. The possible padding pattern for the size
    Rnd is (331-Rnd+1)^2.
    '''
    def __init__(self):
        super(Randomization, self).__init__()
         # 生成一个[299,331)的随机数
        self.resize_shape = np.random.randint(299,331) 
        # 生成一个左边界padding的随机数
        self.padding_left = np.random.randint(0, (331 - self.resize_shape+1))
        # 生成一个上边界padding的随机数
        self.padding_top = np.random.randint(0, (331 - self.resize_shape+1))
        # 计算右边界和下边界  
        self.padding_right = 331 - self.resize_shape - self.padding_left
        self.padding_bottom = 331 - self.resize_shape - self.padding_top


    def forward(self,x):
        # 生成一个随机resize的图像
        x = F.interpolate(x,self.resize_shape)  
        # 随机padding到固定的大小331
        x = F.pad(x, [self.padding_top, self.padding_bottom,self.padding_left, self.padding_right])
        return x

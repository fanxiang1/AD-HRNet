import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01

# 位置注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        # self.bn = BatchNorm2d(1, momentum=BN_MOMENTUM)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        # output = self.bn(output)
        output = self.sigmoid(output)
        return output


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=1):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # nn.BatchNorm2d(channel // reduction,momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            # nn.BatchNorm2d(channel // reduction, momentum=BN_MOMENTUM)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

# 将DAattention和shuffleattention融合
# 做一个可分组的DAattention
class ShuffleDualAttention(nn.Module):

    def __init__(self,channel=512,G=8):
        super().__init__()
        self.G = G
        self.channel = channel

        # 传入两个分支模块的通道是分组再做split之后的通道数
        self.ca = ChannelAttention(channel=int(channel/(2*G)), reduction=1)
        self.sa = SpatialAttention(kernel_size=7)
        # self.position_attention_module = SpatialAttention(kernel_size=7)
        # self.channel_attention_module = ChannelAttention(int(channel/(2*G)))

    # 通道混合
    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        residual = x

        # 开始对特征进行分组
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w
        # print(x.shape)

        # 进行通道分离
        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w
        bs, c, h, w = x_0.shape

        # 位置注意力
        # print(x_0.shape)
        p_out = x_0*self.sa(x_0)
        # print(p_out.shape)
        # 通道注意力
        c_out = x_1*self.ca(x_1)
        # print(c_out.shape)
        # reshape成原来的格式
        # p_out = p_out.permute(0, 2, 1).view(bs, c, h, w)
        # print(p_out.shape) # [400,32,7,7]
        # c_out = c_out.view(bs, c, h, w)
        # print(c_out.shape) # [400,32,7,7]
        # 在通道维度进行相连
        out = torch.cat([p_out, c_out], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)
        # print(out.shape)

        # 通道shuffle
        out = self.channel_shuffle(out,2)
        return out+residual






if __name__ == '__main__':
    input = torch.randn(1, 72,128, 128)
    danet = ShuffleDualAttention(channel=72,G=9)
    output=danet(input)
    print(output.shape)
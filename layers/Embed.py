import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
import math


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        #使用长度为 3 的卷积核（小的权重矩阵），每次在输入序列的 3 个相邻元素上滑动，并计算它们的加权和，然后输出一个特征值。这种滑动操作有助于捕捉局部的时间特征或模式。
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, #输入序列的特征维度 -> 嵌入后的特征维度  都是16；每个patch长16，conv1后还是16
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False) #使用循环填充模式 (可以在处理时间序列数据时保持序列的周期性特征。)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu') #权重初始化采用 Kaiming 正态分布，适用于使用 Leaky ReLU 激活函数的网络结构

    def forward(self, x):#(batch_size * n_vars, num_patches, patch_len)
        #permute : 将形状从 (batch_size, seq_len, c_in) 变为 (batch_size, c_in, seq_len)，以符合 nn.Conv1d 的输入要求; c_in 是 patch_len
        #transpose : 再次调整维度顺序，将形状从 (batch_size, d_model, seq_len) 变为 (batch_size, seq_len, d_model)，以符合后续处理的要求
        # x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        x = self.tokenConv(x)
        return x


class ReplicationPad1d(nn.Module): #通过复制序列中的最后一个值来进行填充
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding #tuple(0,stride = 8)

    def forward(self, input: Tensor) -> Tensor: #batch, seq_len, emb_dim
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1]) #将这个最后一个元素在新维度上重复 self.padding[-1] 次，生成填充内容。这里的 self.padding[-1] 控制重复的次数，也就是要填充多少个值。
        output = torch.cat([input, replicate_padding], dim=-1) #将（32,1,96） concat (32,1,8) ——input最后那个复制了stride次，确保patch切割时，最后一个一定会被取到
        return output #32，1，104   int((configs.seq_len - self.patch_len) / self.stride + 2) # （96 - 16） / 8 + 2


# patch_len=36=d_model; stride=18; seq_len=5000; patch_nums=277
# patch_len=32=d_model; stride=16; seq_len=5000; patch_nums=312
class PatchEmbedding(nn.Module):
    # def __init__(self, d_model, patch_len, stride, dropout):
    def __init__(self, n_vars, patch_len, stride, dropout):
        """原本是将单变量时序切片成patch_len=16，然后用Conv1d将16维(其实是单变量连续16个时间戳的片段)--卷积-->d_model=32"""
        """现在换成：将channel 12 多变量时序，用Conv1d转为1维，再切片成300patches"""
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride

        self.combine_embedding = TokenEmbedding(n_vars,1)  # 1DCNN insize: patch_len 16 -> outsize: d_model 16

        self.value_embedding = TokenEmbedding(self.patch_len, self.patch_len)

        # 通过复制序列中的最后一个值来进行填充
        self.padding_patch_layer = ReplicationPad1d((0, stride)) # stride:repeat的次数

        # 每个 patch 使用 TokenEmbedding 进行特征映射(1DCNN)，将每个 patch 映射到 d_model 维度。
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): #32，1，96
        """先向右padding，再patch分割，再nn.Conv1d"""
        """改成：先nn.Conv1d (先把12channel压缩成1个)，向右padding，再patch分割"""


        # 每个 patch 使用 TokenEmbedding 进行特征映射(1DCNN)，将每个 patch 映射到 d_model 维度。
        # 12 leads -> 1
        x_reduced = self.combine_embedding(x) #（5，1，5000）
        n_vars = x_reduced.shape[1] #1个变量 （5，12，5000）12个变量
        x = self.padding_patch_layer(x_reduced)  # 32，1，104 #（5，1，+stride）

        # unfold 操作将输入 x 划分为大小为 patch_len 的块（patch）。
        # dimension=-1 表示我们在序列的时间维度上进行操作，
        # size=self.patch_len 是每个 patch 的长度，
        # step=self.stride 是滑动步长。
        # 该操作会返回一个新的张量，表示多个 patch。
        # (batch_size, n_vars, num_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) #切分成12个patch，每个patch长16 (32,1,12,16) #（5，1，patch_nums, patch_len）

        #(batch_size * n_vars, num_patches, patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) #(32,12,16)

        # x = self.value_embedding(x.permute(0, 2, 1)).transpose(1, 2) #channel要放在中间

        return self.dropout(x), x_reduced, n_vars
        # return self.dropout(x), n_vars




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


embedding = nn.Embedding(10, 3)
print(embedding)
# Embedding(10, 3)
print(type(embedding))
# <class 'torch.nn.modules.sparse.Embedding'>
input1 = torch.LongTensor([[1,2,3,4],[5,6,7,8]])
print(embedding(input1))
'''
tensor([[[-0.6906,  1.3950,  0.9294],
         [ 0.1486,  1.6036,  0.5980],
         [ 1.4208,  0.6906, -0.2799],
         [-0.2313, -0.5917, -0.1981]],

        [[-0.2215,  1.7515,  1.2470],
         [ 1.0987, -0.2065, -0.1494],
         [-1.0551, -0.5416,  0.0256],
         [-0.0311, -0.5572, -0.2476]]], grad_fn=<EmbeddingBackward0>)
'''
embedding = nn.Embedding(10, 3, padding_idx = 0)
input1 = torch.LongTensor([[2, 5, 8, 3]])
print(embedding(input1))
'''
tensor([[[ 0.5118,  0.2931,  1.9410],
         [ 1.0358,  0.7886, -0.2937],
         [-0.0711,  0.5964, -0.8384],
         [-0.5199, -2.0837,  0.0347]]], grad_fn=<EmbeddingBackward0>)
'''
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        # d_model: 词嵌入的维度
        # vocab: 词表的大小
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        # 将参数传入类中
        self.d_model = d_model

    def forward(self, x):
        # x:代表输入进模型的文本通过词汇映射后的数字张量
        return self.lut(x) * math.sqrt(self.d_model)

d_model= 512
vocab = 1000
x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
emb = Embeddings(d_model, vocab)
embr = emb(x)
print("embr:", embr)
print(embr.shape)
'''
embr: tensor([[[-21.5972,  28.0914,   1.6477,  ...,  -2.5906,  21.6872, -15.1827],
         [ 12.2383, -12.2544,   6.5127,  ..., -22.4092, -22.3950,  16.0831],
         [-10.8122,  -6.9863,   6.4863,  ...,  -7.1786,  -7.9925,   6.9803],
         [ 30.6013,  17.8523,   8.3321,  ...,   6.6138,   9.1388,  32.3414]],

        [[ -7.3983,  20.0720,   6.5595,  ...,  37.4633,  27.4935, -13.5691],
         [-13.0629,  38.7053, -45.7595,  ..., -16.5600,   2.2317,  21.4354],
         [ -6.2126,  14.3673, -42.1971,  ...,  34.6477, -10.8796,  -4.4500],
         [  0.4145,  20.5080, -10.5656,  ...,  13.8754,  12.2417,  14.2898]]],
       grad_fn=<MulBackward0>)
torch.Size([2, 4, 512])
'''
'''
这个文件主要是学习embedding这个模块的实现类，实例参考学习
'''

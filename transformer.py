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


m = nn.Dropout(p = 0.2)
input1 = torch.randn(4, 5)
print(input1)
output = m(input1)
print(output)
'''
tensor([[-0.2901,  0.0307,  1.2515,  1.0127,  1.4962],
        [ 2.1598, -0.3480, -0.6157,  0.1837,  0.0137],
        [-0.5211, -1.0049, -1.8923, -0.8710, -0.4527],
        [ 1.2412,  2.2449, -0.8510, -0.0733,  0.1400]])
tensor([[-0.3626,  0.0384,  1.5644,  1.2659,  1.8702],
        [ 0.0000, -0.4350, -0.7696,  0.2297,  0.0172],
        [-0.6514, -1.2561, -2.3654, -1.0888, -0.5659],
        [ 1.5515,  0.0000, -1.0638, -0.0917,  0.1750]])
'''
x = torch.tensor([1, 2, 3, 4])
y = torch.unsqueeze(x, 0)
print(y)
z = torch.unsqueeze(x, 1)
print(z)
'''
tensor([[1, 2, 3, 4]])
tensor([[1],
        [2],
        [3],
        [4]])
'''


# 构建位置编码器的类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        # d_model: 代表词嵌入的维度
        # dropout: 代表Dropout层的置零比率
        # max_len: 代表每隔句子的最大长度
        super(PositionalEncoding, self).__init__()

        # 实例化Dropout层
        self.dropout = nn.Dropout(p = dropout)
        
        # 初始化一个位置编码矩阵，大小是max_len * d_model
        pe = torch.zeros(max_len, d_model)
        
        # 初始化一个绝对位置矩阵，max_len * 1
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # 定义一个变化矩阵div_term,跳跃式的初始化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
   
        # 将前面定义的变化矩阵进行奇数、偶数的分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将二维张量扩充成三维张量
        pe = pe.unsqueeze(0)

        # 将位置矩阵注册成模型的buffer,这个buffer不是模型中的参数，不跟随优化器同步更新
        # 注册成buffer后可以在模型保存后重新加载的时候，将这个位置编码器和模型参数一同加载进去
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x:代表文本序列的词嵌入表示
        # 首先明确pe的编码太长了，将第二个维度，就是max_len对应的那个维度缩小成x的句子
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

d_model = 512
dropout = 0.1
max_len = 60

x = embr
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
print(pe_result)
print(pe_result.shape)
'''
tensor([[[-23.9969,  32.3238,   1.8308,  ...,  -1.7674,  24.0969, -15.7586],
         [ 14.5331, -13.0156,   8.1495,  ..., -23.7880, -24.8832,  18.9812],
         [-11.0032,  -8.2249,   8.2474,  ...,  -6.8651,  -8.8804,   8.8670],
         [ 34.1582,  18.7359,   9.5302,  ...,   8.4597,  10.1546,  37.0460]],

        [[ -8.2203,  23.4133,   7.2883,  ...,  42.7370,   0.0000,  -0.0000],
         [-13.5794,  43.6063, -49.9307,  ..., -17.2889,   2.4798,  24.9282],
         [ -5.8925,  15.5012,  -0.0000,  ...,  39.6086, -12.0882,  -3.8333],
         [  0.6173,  21.6867, -11.4672,  ...,  16.5282,   0.0000,  16.9887]]],
       grad_fn=<MulBackward0>)
torch.Size([2, 4, 512])
'''


# 先设置画布
plt.figure(figsize = (15, 5))

# 实例化PositionalEncoding类对象，词嵌入维度为20，置零比率设置为0
pe = PositionalEncoding(20, 0)

# 往pe中传入一个全零初始化的变量x，相当于展示pe
y = pe(Variable(torch.zeros(1, 100, 20)))

plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())

plt.legend(["dim %d" %p for p in [4, 5, 6, 7]]) 


print(np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], k = -1))
print(np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], k = 0))
print(np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], k = 1))


'''
[[ 1  2  3]
 [ 4  5  6]
 [ 0  8  9]
 [ 0  0 12]]
[[1 2 3]
 [0 5 6]
 [0 0 9]
 [0 0 0]]
[[0 2 3]
 [0 0 6]
 [0 0 0]
 [0 0 0]]
'''

# 构建掩码张量函数
def subsequent_mask(size):
    # size：代表掩码张量最后两个维度，形成一个方阵
    attn_shape = (1, size, size)

    # 使用np.ones()先构建一个全1的张量，然后利用np.triu()形成上三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')

    # 使得这个三角矩阵反转
    return torch.from_numpy(1 - subsequent_mask)

size = 5
sm = subsequent_mask(size)
# print("sm:", sm)

plt.figure(figsize = (5, 5))
plt.imshow(subsequent_mask(20)[0])



x = Variable(torch.randn(5, 5))
print(x)

mask = Variable(torch.zeros(5, 5))
print(mask)

y = x.masked_fill(mask == 0, 1e-9)
print(y)

'''
tensor([[ 0.1921, -1.3656,  0.9082, -0.9860,  1.4314],
        [ 0.9230,  0.8740, -0.0495,  0.6527,  1.1498],
        [-0.4922, -0.3249, -0.1321, -0.1768,  1.7603],
        [-0.0906, -0.9663,  1.1037, -0.1717, -2.3737],
        [ 1.4549,  0.3206,  0.4819,  0.3243, -1.0052]])
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
tensor([[1.0000e-09, 1.0000e-09, 1.0000e-09, 1.0000e-09, 1.0000e-09],
        [1.0000e-09, 1.0000e-09, 1.0000e-09, 1.0000e-09, 1.0000e-09],
        [1.0000e-09, 1.0000e-09, 1.0000e-09, 1.0000e-09, 1.0000e-09],
        [1.0000e-09, 1.0000e-09, 1.0000e-09, 1.0000e-09, 1.0000e-09],
        [1.0000e-09, 1.0000e-09, 1.0000e-09, 1.0000e-09, 1.0000e-09]])
'''


def attention(query, key, value, mask=None, dropout=None):
    # query, key, value:代表注意力的三个输入张量
    # mask: 掩码张量
    # dropout: 传入的dropout实例化对象
    # 首先把query的最后一个维度提取出来，代表词嵌入的维度
    d_k = query.size(-1)

    # 根据注意力计算公式，把query和key的转置进行矩阵乘法，然后除以缩放稀疏
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 判断是否使用掩码张量
    if mask is not None:
        # 利用masked_fill方法，将掩码张量和0进行位置的意义比较，如果掩码等于0，替换成一个非常小的数值
        scores = score.masked_fill(mask == 0, 1e-9)
    
    # 对scores最后一个维度进行sofrmask操作，得到注意力矩阵
    p_attn = F.softmax(scores, dim=-1)

    # 判断是否使用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 最后一步完成p_attn和value张量的乘法，并返回query注意力表示
    return torch.matmul(p_attn, value), p_attn

query = key = value = pe_result
print(pe_result)
attn, p_attn = attention(query, key, value)
print("attn:", attn)
print(attn.shape)
print("p_attn:", p_attn)
print(p_attn.shape)

'''
tensor([[[  0.0000, -10.2716,   0.0000,  ...,  17.5836, -38.6300, -25.5580],
         [ 41.6243,   0.0000, -27.6595,  ...,  -2.1759,  18.1972, -22.1183],
         [ 15.1254,  33.7136, -27.9896,  ..., -30.8425, -30.4634,   6.3921],
         [-31.1349,  35.1394,  36.4696,  ...,   0.6742,  15.7294, -31.1496]],

        [[-10.1095,  18.7192,  23.7893,  ...,  24.9475, -20.4034, -50.5805],
         [ 14.3130,   6.8231,  -0.0000,  ...,  15.5989,  -3.5088,  62.9959],
         [ 15.0196, -28.7547,  -6.6247,  ...,  22.5821, -43.1670,  59.9169],
         [-19.8802,  15.2179,   7.8016,  ..., -10.9695, -13.8674,   0.0000]]],
       grad_fn=<MulBackward0>)
attn: tensor([[[  0.0000, -10.2716,   0.0000,  ...,  17.5836, -38.6300, -25.5580],
         [ 41.6243,   0.0000, -27.6595,  ...,  -2.1759,  18.1972, -22.1183],
         [ 15.1254,  33.7136, -27.9896,  ..., -30.8425, -30.4634,   6.3921],
         [-31.1349,  35.1394,  36.4696,  ...,   0.6742,  15.7294, -31.1496]],

        [[-10.1095,  18.7192,  23.7893,  ...,  24.9475, -20.4034, -50.5805],
         [ 14.3130,   6.8231,   0.0000,  ...,  15.5989,  -3.5088,  62.9959],
         [ 15.0196, -28.7547,  -6.6247,  ...,  22.5821, -43.1670,  59.9169],
         [-19.8802,  15.2179,   7.8016,  ..., -10.9695, -13.8674,   0.0000]]],
       grad_fn=<UnsafeViewBackward0>)
torch.Size([2, 4, 512])
p_attn: tensor([[[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.]],

        [[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.]]], grad_fn=<SoftmaxBackward0>)
torch.Size([2, 4, 4])

'''

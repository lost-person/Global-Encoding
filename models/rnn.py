import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models
import math
import numpy as np

class rnn_encoder(nn.Module):
    """
    encoder定义，seq2seq调用它，传递config过来
    """
    def __init__(self, config, embedding=None):# config为模型配置
        super(rnn_encoder, self).__init__()
        # 如果embedding为空，则使用新的embedding
        self.embedding = embedding if embedding is not None else nn.Embedding(config.src_vocab_size, config.emb_size)
        self.hidden_size = config.hidden_size
        self.config = config
        # True，加入卷积层
        if config.swish:
            # Conv参数：输入通道数（词向量维度）、产生通道数、卷积核大小
            self.sw1 = nn.Sequential(nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, padding=0), nn.BatchNorm1d(config.hidden_size), nn.ReLU())
            self.sw3 = nn.Sequential(nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm1d(config.hidden_size),
                                     nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(config.hidden_size))
            self.sw33 = nn.Sequential(nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm1d(config.hidden_size),
                                      nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(config.hidden_size),
                                      nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(config.hidden_size))
            # 线性连接单元，在rnn层过后，输出双向拼接
            self.linear = nn.Sequential(nn.Linear(2*config.hidden_size, 2*config.hidden_size), nn.GLU(), nn.Dropout(config.dropout))
            # 将3d变为d
            self.filter_linear = nn.Linear(3*config.hidden_size, config.hidden_size)
            self.tanh = nn.Tanh()
            self.sigmoid = nn.Sigmoid()
        # 自注意力机制
        if config.selfatt:
            if config.attention == 'None':
                self.attention = None
            elif config.attention == 'bahdanau':
                self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size, config.pool_size)
            elif config.attention == 'luong':
                self.attention = models.luong_attention(config.hidden_size, config.emb_size, config.pool_size)
            # 配置中使用该注意力机制
            elif config.attention == 'luong_gate':
                self.attention = models.luong_gate_attention(config.hidden_size, config.emb_size)
        # encoder中GRU或LSTM层定义，默认双向LSTM，3层
        if config.cell == 'gru':
            self.rnn = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                              num_layers=config.enc_num_layers, dropout=config.dropout,
                              bidirectional=config.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                               num_layers=config.enc_num_layers, dropout=config.dropout,
                               bidirectional=config.bidirectional)

    def forward(self, inputs, lengths):
        """
        前馈计算，参数为输入文本id表示(一个batch，矩阵表示)以及文本长度，返回encoder输出和状态
        """
        # pack为torch.nn.utils.rnn.pack_padded_sequence()，前面的排序是为此操作，返回一个PackedSequence 对象。功能是将一个填充后的变长序列压紧。
        embs = pack(self.embedding(inputs), lengths)
        # rnn输出为输出h和最后时刻隐状态G
        outputs, state = self.rnn(embs)
        # unpack为torch.nn.utils.rnn.pad_packed_sequence()，与pack相反，把压紧的序列再填充回来。填充时会初始化为0。
        outputs = unpack(outputs)[0] #len * batch_size * 2h_size(双向进行了拼接)
        if self.config.bidirectional: # True
            if self.config.swish: # TRUE
                outputs = self.linear(outputs)# 结果通过线性层输出len * batch_size * 2h_size [127，64，512]（因为过了GLU）
            else:
                outputs = outputs[:,:,:self.config.hidden_size] + outputs[:,:,self.config.hidden_size:]
        if self.config.swish:
            # 进行卷积操作
            outputs = outputs.transpose(0,1).transpose(1,2) # [64，512，127]
            conv1 = self.sw1(outputs) # 维度不变[64，512，127]
            conv3 = self.sw3(outputs)
            conv33 = self.sw33(outputs)
            conv = torch.cat((conv1, conv3, conv33), 1) # 按列拼接 [64, 1536, 127]
            conv = self.filter_linear(conv.transpose(1,2)) # 3*h_size->h_size [64，127，512]
            if self.config.selfatt: # True
                conv = conv.transpose(0,1) # [127，64，512]
                outputs = outputs.transpose(1,2).transpose(0,1) # [127，64，512]
            else:
                gate = self.sigmoid(conv)
                outputs = outputs * gate.transpose(1,2)
                outputs = outputs.transpose(1,2).transpose(0,1)
        if self.config.selfatt: # True
            self.attention.init_context(context=conv) # 进行了维度转换
            out_attn, weights = self.attention(conv, selfatt=True) # 返回自注意力机制结果输出和权重矩阵
            gate = self.sigmoid(out_attn) # Length * Batch_size * Hidden_size
            outputs = outputs * gate # 卷积结果点乘上自注意力输出
        if self.config.cell == 'gru':
            state = state[:self.config.dec_num_layers]
        else:# state[0].size() [6,batch_size,hidden_size],6代表三层双向，但在这里每层只取了前向的结果？？？？？？？？？？？？？？？？？？？？？？？？？
            state = (state[0][::2], state[1][::2])# 最后时刻的隐层状态h，细胞单元状态c，m::n，m表示从下标m开始，n为步长，相当于maxpooling（2）操作得到state[0].size() [3,batch_size,hidden_size]

        return outputs, state

class rnn_decoder(nn.Module):

    def __init__(self, config, embedding=None, use_attention=True):
        super(rnn_decoder, self).__init__()
        # 已经embbeding过，使用encoder的emb
        self.embedding = embedding if embedding is not None else nn.Embedding(config.tgt_vocab_size, config.emb_size)

        input_size = config.emb_size

        if config.cell == 'gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.dec_num_layers, dropout=config.dropout)
        else:# 三层单向LSTM
            self.rnn = StackedLSTM(input_size=input_size, hidden_size=config.hidden_size,
                                   num_layers=config.dec_num_layers, dropout=config.dropout)

        self.linear = nn.Linear(config.hidden_size, config.tgt_vocab_size)
        self.linear_ = nn.Linear(config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        if not use_attention or config.attention == 'None':
            self.attention = None
        elif config.attention == 'bahdanau':
            self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong':
            self.attention = models.luong_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong_gate':
            self.attention = models.luong_gate_attention(config.hidden_size, config.emb_size, prob=config.dropout)

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

        if config.pointer:
            self.p_linear = nn.Linear(config.hidden_size * 2 + config.emb_size, 1)

    def forward(self, input, state):
        # 获取前一时刻输出
        embs = self.embedding(input)
        output, state = self.rnn(embs, state)# 定义为栈式LSTM
        if self.attention is not None:
            if self.config.attention == 'luong_gate':
                context, attn_weights = self.attention(output) # 进行att，output和context(卷积的输出)
            else:
                context, attn_weights = self.attention(output, embs)
        else:
            attn_weights = None
        # 计算得分，输出词表大小
        output = self.compute_score(context)
        p = None
        if self.config.pointer:
            p = self.sigmoid(self.p_linear(torch.cat((context, state[0][2], embs), 1)))
            output = p * output

        return output, state, attn_weights, p

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList() # 设计用来存储任意数量的nn.module。
        # 三层
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):# input为当前时刻emb（64，512）
        h_0, c_0 = hidden #hidden为encoder最终输出
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            # 过LSTM层，每层使用encoder每层的h和c
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i])) # 维度均为[64,512]
            input = h_1_i
            if i + 1 != self.num_layers:# 不是最后一层便使用dropout
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]
        # 获得下一时刻的隐状态和细胞状态
        h_1 = torch.stack(h_1) # 默认dim=0，[3,64,512]
        c_1 = torch.stack(c_1)
        # 返回三层输出结果
        return input, (h_1, c_1)


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1

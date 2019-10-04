# encoding=utf-8
import linecache
import torch
import torch.utils.data as torch_data
from random import Random
import utils

num_samples = 1


class MonoDataset(torch_data.Dataset):

    def __init__(self, infos, indexes=None):

        self.srcF = infos['srcF']
        self.original_srcF = infos['original_srcF']
        self.length = infos['length']
        self.infos = infos
        if indexes is None:
            self.indexes = list(range(self.length))
        else:
            self.indexes = indexes

    def __getitem__(self, index):
        index = self.indexes[index]
        src = list(map(int, linecache.getline(self.srcF, index+1).strip().split()))
        original_src = linecache.getline(self.original_srcF, index+1).strip().split()

        return src, original_src

    def __len__(self):
        return len(self.indexes)


class BiDataset(torch_data.Dataset):

    def __init__(self, infos, indexes=None, char=False, copy=False):

        self.srcF = infos['srcF']
        self.tgtF = infos['tgtF']
        self.original_srcF = infos['original_srcF']
        self.original_tgtF = infos['original_tgtF']
        self.length = infos['length']
        self.infos = infos
        self.char = char
        self.copy = copy
        if copy:
            self.srcoovF = infos['srcoovF']
            self.tgtoovF = infos['tgtoovF']
            self.oovs = infos['oovF']
        if indexes is None:
            self.indexes = list(range(self.length))
        else:
            self.indexes = indexes

    def __getitem__(self, index):
        # 在遍历训练数据时使用
        index = self.indexes[index]
        src = list(map(int, linecache.getline(self.srcF, index + 1).strip().split()))
        tgt = list(map(int, linecache.getline(self.tgtF, index + 1).strip().split()))
        original_src = linecache.getline(self.original_srcF, index + 1).strip().split()
        original_tgt = linecache.getline(self.original_tgtF, index + 1).strip().split() if not self.char else \
                        list(linecache.getline(self.original_tgtF, index + 1).strip())
        if self.copy:
            src_oov = list(map(int, linecache.getline(self.srcoovF, index+1).strip().split()))
            tgt_oov = list(map(int, linecache.getline(self.tgtoovF, index+1).strip().split()))
            oovs = list(linecache.getline(self.oovs, index+1).strip())

            return src, tgt, original_src, original_tgt, src_oov, tgt_oov, oovs
        return src, tgt, original_src, original_tgt

    def __len__(self):
        return len(self.indexes)


def splitDataset(data_set, sizes):
    length = len(data_set)
    indexes = list(range(length))
    rng = Random()
    rng.seed(1234)
    rng.shuffle(indexes)

    data_sets = []
    part_len = int(length / sizes)
    for i in range(sizes-1):
        data_sets.append(BiDataset(data_set.infos, indexes[0:part_len]))
        indexes = indexes[part_len:]
    data_sets.append(BiDataset(data_set.infos, indexes))
    return data_sets


def padding(data):
    """
    分别对src和tgt进行padding进行padding
    :param data: 输入数据
    :return:
    """
    copy = False
    if len(data[0])==7:
        copy = True

    # 将tuple组成的list拆分成多个tuple:即 将id与文本分离
    if copy:
        src, tgt, original_src, original_tgt, src_oov, tgt_oov, oovs = zip(*data)
    else:
        src, tgt, original_src, original_tgt = zip(*data)
    # 获取文本长度
    src_len = [len(s) for s in src]
    tgt_len = [len(s) for s in tgt]
    # 构建一个0矩阵，每个句子对应大小为max(src_len)，行数为batch_size
    src_pad = torch.zeros(len(src), max(src_len)).long()
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    if copy:
        # max_oov = max([len(v) for v in oovs])
        # print(max_oov)
        # oov_pad = torch.zeros(len(src), max_oov)
        # print(oov_pad)
        srcoov_pad = torch.zeros(len(src), max(src_len)).long()
        tgtoov_pad = torch.zeros(len(tgt), max(tgt_len)).long()
        for i, s in enumerate(src_oov):
            end = src_len[i]
            # srcoov_pad[i, :end] = torch.LongTensor(s[end - 1::-1])
            srcoov_pad[i, :end] = torch.LongTensor(s)[:end]
        for i, s in enumerate(tgt_oov):
            end = tgt_len[i]
            tgtoov_pad[i, :end] = torch.LongTensor(s)[:end]  # 并未进行翻转
        oovs = list(oovs)
    for i, s in enumerate(src):
        end = src_len[i]
        # src_pad[i, :end] = torch.LongTensor(s[end-1::-1]) # 取从下标为end-1的元素，翻转读取，即翻转取整个句子
        src_pad[i, :end] = torch.LongTensor(s)[:end]
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)[:end] # 并未进行翻转
    if copy:
        return src_pad, tgt_pad, \
           torch.LongTensor(src_len), torch.LongTensor(tgt_len), \
           original_src, original_tgt, srcoov_pad, tgtoov_pad, oovs
    return src_pad, tgt_pad, \
           torch.LongTensor(src_len), torch.LongTensor(tgt_len), \
           original_src, original_tgt

def ae_padding(data):
    src, tgt, original_src, original_tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s)[:end]

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]

        tgt_pad[i, :end] = torch.LongTensor(s)[:end]

    ae_len = [len(s)+2 for s in src]
    ae_pad = torch.zeros(len(src), max(ae_len)).long()
    for i, s in enumerate(src):
        end = ae_len[i]
        ae_pad[i, 0] = utils.BOS
        ae_pad[i, 1:end-1] = torch.LongTensor(s)[:end-2]
        ae_pad[i, end-1] = utils.EOS

    return src_pad, tgt_pad, ae_pad, \
           torch.LongTensor(src_len), torch.LongTensor(tgt_len), torch.LongTensor(ae_len), \
           original_src, original_tgt


def split_padding(data):
    src, tgt, original_src, original_tgt = zip(*data)

    split_samples = []
    num_per_sample = int(len(src) / utils.num_samples)

    for i in range(utils.num_samples):
        split_src = src[i*num_per_sample:(i+1)*num_per_sample]
        split_tgt = tgt[i*num_per_sample:(i+1)*num_per_sample]
        split_original_src = original_src[i * num_per_sample:(i + 1) * num_per_sample]
        split_original_tgt = original_tgt[i * num_per_sample:(i + 1) * num_per_sample]

        src_len = [len(s) for s in split_src]
        src_pad = torch.zeros(len(split_src), max(src_len)).long()
        for i, s in enumerate(split_src):
            end = src_len[i]
            src_pad[i, :end] = torch.LongTensor(s)[:end]

        tgt_len = [len(s) for s in split_tgt]
        tgt_pad = torch.zeros(len(split_tgt), max(tgt_len)).long()
        for i, s in enumerate(split_tgt):
            end = tgt_len[i]
            tgt_pad[i, :end] = torch.LongTensor(s)[:end]

        split_samples.append([src_pad, tgt_pad,
                              torch.LongTensor(src_len), torch.LongTensor(tgt_len),
                              split_original_src, split_original_tgt])

    return split_samples
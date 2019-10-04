import torch
import pickle

def load_data():
    """
    作用：加载数据
    参数：无
    返回值：无
    """
    print('loading data...\n')
    # 加载preprocess过的data（包括train、valid、test数据以及词表）
    data = pickle.load(open(config.data+'data.pkl', 'rb'))
    # scale为训练集的比例，默认为1
    data['train']['length'] = int(data['train']['length'] * opt.scale)

    # 获取训练集和验证集对象
    trainset = utils.BiDataset(data['train'], char=config.char)
    validset = utils.BiDataset(data['valid'], char=config.char)

    if opt.pointer:
        trainset = utils.BiDataset(data['train'], char=config.char, copy=opt.pointer)
        validset = utils.BiDataset(data['valid'], char=config.char, copy=opt.pointer)

    # 获取词表及其大小，不用改
    src_vocab = data['dict']['src']
    tgt_vocab = data['dict']['tgt']
    config.src_vocab_size = src_vocab.size()
    config.tgt_vocab_size = tgt_vocab.size()

    # 获取一个batch的data，将每个batch的数据长度取最大，不够长度的padding0
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              collate_fn=utils.padding)
    # 如果没有单独设置valid的batch大小，则使用与train相同大小batch
    if hasattr(config, 'valid_batch_size'):
        valid_batch_size = config.valid_batch_size
    else:
        valid_batch_size = config.batch_size
    validloader = torch.utils.data.DataLoader(dataset=validset,
                                              batch_size=valid_batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              collate_fn=utils.padding)
    if opt.pointer:
        trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                                  batch_size=config.batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  collate_fn=utils.padding)
        validloader = torch.utils.data.DataLoader(dataset=validset,
                                                  batch_size=valid_batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  collate_fn=utils.padding)
    return {'trainset': trainset, 'validset': validset,
            'trainloader': trainloader, 'validloader': validloader,
            'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}
import torch
import torch.nn as nn
import torch.utils.data
import lr_scheduler as L

import os
import argparse
import pickle
import time
from collections import OrderedDict

import opts
import models
import utils
import codecs
from rouge import Rouge

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-pointer', action='store_true', help='add the copy mechanism')
parser.add_argument('-rl', action='store_true', help='add the reinforce learning')
# 使用opts文件参数配置
opts.model_opts(parser)
# 调用 parse_args() 来解析程序的命令行，返回保存有命令行参数值的opt对象
opt = parser.parse_args()

# 调用read_config函数，传入参数opt.config(为配置文件yaml)，返回AttrDict对象
config = utils.read_config(opt.config)

# 手动加入yaml参数
config['data'] = './LCSTS_ORIGIN/res/zl.'
config['logF'] = './experiments/lcsts/'
config['epoch']=20
config['batch_size']=64
config['optim']='adam'
config['cell']='lstm'
config['attention']='luong_gate'
config['learning_rate']=0.0003
config['max_grad_norm']=10
config['learning_rate_decay']=0.5
config['start_decay_at']=6
config['emb_size']=512
config['hidden_size']=512
config['dec_num_layers']=3
config['enc_num_layers']=3
config['bidirectional']=True
config['dropout']=0.0
config['max_time_step']=50
config['eval_interval']=10000
config['save_interval']=3000
config['metrics']=['rouge']
config['shared_vocab']=True
config['beam_size']=10
config['unk']=True
config['schedule']=False
config['selfatt']=True
config['schesamp']=False
config['swish']=True
config['length_norm']=True
config['alpha']=0.3
torch.manual_seed(opt.seed)
# 将opt中的参数加入config
opts.convert_to_config(opt, config)
# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
config.use_cuda = use_cuda


if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True

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

def build_model(checkpoints, print_log):
    """
    作用：创建模型
    参数：checkpoints、print_log:
    返回值：model, optim, print_log
    """
    # 将配置参数写入log文件
    # for k, v in config.items():
    #     print_log("%s:\t%s\n" % (str(k), str(v)))
    # --------------创建模型------------- #
    print('building model...\n')
    # getattr() 函数用于返回一个对象属性值。返回一个seq2seq对象, config 作为模型对象初始化参数
    model = getattr(models, opt.model)(config)
    if checkpoints is not None:
        # 加载模型参数
        model.load_state_dict(checkpoints['model'])
    if opt.pretrain:
        # 加载预训练
        print('loading checkpoint from %s' % opt.pretrain)
        pre_ckpt = torch.load(opt.pretrain)['model']
        pre_ckpt = OrderedDict({key[8:]: pre_ckpt[key] for key in pre_ckpt if key.startswith('encoder')})
        print(model.encoder.state_dict().keys())
        print(pre_ckpt.keys())
        model.encoder.load_state_dict(pre_ckpt)
    if use_cuda:
        model.cuda()
    
    # optimizer
    if checkpoints is not None:
        optim = checkpoints['optim']
    else:
        # 使用Optim对象
        optim = models.Optim(config.optim, config.learning_rate, config.max_grad_norm,
                             lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
    # 设置optim的参数
    optim.set_parameters(model.parameters())

    # print log，并将模型参数打印出来
    # param_count = 0
    # for param in model.parameters():
    #     param_count += param.view(-1).size()[0]
    # for k, v in config.items():
    #     print_log("%s:\t%s\n" % (str(k), str(v)))
    # print_log("\n")
    # print_log(repr(model) + "\n\n")
    # print_log('total number of parameters: %d\n\n' % param_count)

    return model, optim, print_log

def train_model(model, data, optim, epoch, params):
    """
    模型训练函数
    :param model: 模型
    :param data: 数据
    :param optim: 优化器
    :param epoch: 第i个epoch
    :param params: 模型参数，字典形式，包括rouge
    :return: 无
    """
    tgt_vocab = data['tgt_vocab']
    # 表示当前为训练模式
    model.train()
    # 读取batch数据
    trainloader = data['trainloader']
    if opt.pointer:
        # 获取源文本（id表示，0padding）、目标文本、源文本一个batch中每个记录的长度、目标文本长度、original_tgt表示文本表示
        for src, tgt, src_len, tgt_len, original_src, original_tgt, srcoov_pad, tgtoov_pad, oovs in trainloader:
            model.zero_grad()

            if config.use_cuda:
                # 将内存中的数据复制到GPU的显存中计算
                src = src.cuda()
                srcoov_pad = srcoov_pad.cuda()
                tgtoov_pad = tgtoov_pad.cuda()
                oovs = oovs.cuda()
                tgt = tgt.cuda()
                src_len = src_len.cuda()
            # 对src_len降序排序
            lengths, indices = torch.sort(src_len, dim=0, descending=True)
            # 按行进行切片，indices为顺序，即根据排序结果下标排列数据
            src = torch.index_select(src, dim=0, index=indices)
            tgt = torch.index_select(tgt, dim=0, index=indices)
            srcoov_pad = torch.index_select(srcoov_pad, dim=0, index=indices)
            tgtoov_pad = torch.index_select(tgtoov_pad, dim=0, index=indices)

            oov_list = []
            for i in indices:
                oov_list.append(oovs[i])

            # :-1表示除了最后一个取全部，1:表示除了第一个取全部
            dec = tgt[:, :-1]
            targets = tgt[:, 1:]
            try:
                if config.schesamp:  # false
                    if epoch > 8:
                        e = epoch - 8
                        loss, outputs = model(src, lengths, dec, targets, srcoov_pad=srcoov_pad, tgtoov_pad=tgtoov_pad, oov_list=oov_list, teacher_ratio=0.9 ** e)
                    else:
                        loss, outputs = model(src, lengths, dec, targets, srcoov_pad=srcoov_pad, tgtoov_pad=tgtoov_pad, oov_list=oov_list)
                else:
                    loss, outputs = model(src, lengths, dec, targets, srcoov_pad=srcoov_pad, tgtoov_pad=tgtoov_pad, oov_list=oov_list)  # 调用forward，返回交叉熵损失以及输出概率分布，outputs[len, batch, voc_size]

                pred = outputs.max(2)[1]  # 找出概率最大的那个作为预测[len, batch]，取下标
                targets = targets.t()
                num_correct = pred.eq(targets).masked_select(
                    targets.ne(utils.PAD)).sum().item()  # eq函数判断相等返回同等大小矩阵，masked_select进行mask去掉padding，求和计算正确个数
                num_total = targets.ne(utils.PAD).sum().item()  # 总个数
                if config.max_split == 0:
                    loss = torch.sum(loss) / num_total
                    loss.backward()  # 反向传播损失
                optim.step()

                params['report_loss'] += loss.item()
                params['report_correct'] += num_correct
                params['report_total'] += num_total
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
            # 进行模型评估
            utils.progress_bar(params['updates'], config.eval_interval)
            params['updates'] += 1

            if params['updates'] % config.eval_interval == 0:
                params['log']("epoch: %3d, loss: %6.3f, time: %6.3f, updates: %8d, accuracy: %2.2f\n"
                              % (epoch, params['report_loss'], time.time() - params['report_time'],
                                 # 训练之前记录了时间，相减为该interval的时间
                                 params['updates'], params['report_correct'] * 100.0 / params['report_total']))
                print('evaluating after %d updates...\r' % params['updates'])
                score = eval_model(model, data, params)
                for metric in config.metrics:  # 只有rouge
                    params[metric].append(score[metric])
                    if score[metric] >= max(params[metric]):
                        with codecs.open(params['log_path'] + 'best_' + metric + '_prediction.txt', 'w', 'utf-8') as f:
                            f.write(codecs.open(params['log_path'] + 'candidate.txt', 'r', 'utf-8').read())
                        # ckpt保存模型
                        save_model(params['log_path'] + 'best_' + metric + '_checkpoint.pt', model, optim,
                                   params['updates'])
                model.train()
                params['report_loss'], params['report_time'] = 0, time.time()
                params['report_correct'], params['report_total'] = 0, 0
            # 保存模型，save_interval为3000，每3000步保存模型
            if params['updates'] % config.save_interval == 0:
                save_model(params['log_path'] + 'checkpoint.pt', model, optim, params['updates'])
    else:
        # 获取源文本（id表示，0padding）、目标文本、源文本一个batch中每个记录的长度、目标文本长度、original_tgt表示文本表示
        for src, tgt, src_len, tgt_len, original_src, original_tgt in trainloader:
            model.zero_grad()

            if config.use_cuda:
                # 将内存中的数据复制到GPU的显存中计算
                src = src.cuda()
                tgt = tgt.cuda()
                src_len = src_len.cuda()
            # 对src_len降序排序
            lengths, indices = torch.sort(src_len, dim=0, descending=True)
            # 按行进行切片，indices为顺序，即根据排序结果下标排列数据
            src = torch.index_select(src, dim=0, index=indices)
            tgt = torch.index_select(tgt, dim=0, index=indices)
            # :-1表示除了最后一个取全部，1:表示除了第一个取全部
            dec = tgt[:, :-1]
            targets = tgt[:, 1:]
            try:
                if config.schesamp:# false
                    if epoch > 8:
                        e = epoch - 8
                        loss, outputs = model(src, lengths, dec, targets, teacher_ratio=0.9**e)
                    else:
                        loss, outputs = model(src, lengths, dec, targets)
                else:
                    loss, outputs = model(src, lengths, dec, targets)# 调用forward，返回交叉熵损失以及输出概率分布，outputs[len, batch, vocab_size]
                
                pred = outputs.max(2)[1] # 找出概率最大的那个作为预测[len, batch]，取下标
                
                if config.rl:
                    sample_pred = []
                    for i in range(outputs.size(0)):
                        sample_pred.append(torch.multinomial(outputs[i], 1))
                    
                    # 计算 loss
                    criterion = nn.CrossEntropyLoss(ignore_index=utils.PAD, reduction='none')
                    criterion.cuda()
                    sample_loss = [] # [batch_size]
                    for i in range(outputs.size(1)):
                        sample_loss.append(criterion(outputs[:, i, :], sample_pred[:, i]))
                    sample_loss = torch.tensor(sample_loss)
                    
                    # 转换为词
                    pred_sen_list = [" ".join(tgt_vocab.convertToLabels(sen, utils.EOS)) for sen in pred.t()]
                    sample_pred_sen_list = [" ".join(tgt_vocab.convertToLabels(sen, utils.EOS)) for sen in sample_pred.t()]
                    reference = [" ".join(list(ori_tgt[0])) for ori_tgt in original_tgt]
                    
                    # 计算 rouge
                    rouge = Rouge()
                    pred_score_list = rouge.get_scores(pred_sen_list, reference)
                    sample_pred_score_list = rouge.get_scores(sample_pred_sen_list, reference)

                    neg_reward = []
                    for pred_score, sample_score in zip(pred_score_list, sample_pred_score_list):
                        pred_mean_score = 0.5 * (pred_score['rouge-2']['f'] + pred_score['rouge-l']['f'])
                        sample_mean_score = 0.5 * (sample_score['rouge-2']['f'] + sample_score['rouge-l']['f'])
                        neg_reward.append(sample_mean_score - pred_mean_score)
                    
                    neg_reward = torch.tensor(neg_reward)
                    rl_loss = - torch.mul(sample_loss, neg_reward) / config.batch_size

                targets = targets.t()
                num_correct = pred.eq(targets).masked_select(targets.ne(utils.PAD)).sum().item()# eq函数判断相等返回同等大小矩阵，masked_select进行mask去掉padding，求和计算正确个数
                num_total = targets.ne(utils.PAD).sum().item() #总个数
                if config.max_split == 0:
                    loss = torch.sum(loss) / num_total
                    if config.rl:
                        loss += config.alpha * rl_loss
                    loss.backward() # 反向传播损失
                optim.step()

                params['report_loss'] += loss.item()
                params['report_correct'] += num_correct
                params['report_total'] += num_total
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
            # 进行模型评估
            utils.progress_bar(params['updates'], config.eval_interval)
            params['updates'] += 1

            if params['updates'] % config.eval_interval == 0:
                params['log']("epoch: %3d, loss: %6.3f, time: %6.3f, updates: %8d, accuracy: %2.2f\n"
                              % (epoch, params['report_loss'], time.time()-params['report_time'], #训练之前记录了时间，相减为该interval的时间
                                 params['updates'], params['report_correct'] * 100.0 / params['report_total']))
                print('evaluating after %d updates...\r' % params['updates'])
                score = eval_model(model, data, params)
                for metric in config.metrics: #只有rouge
                    params[metric].append(score[metric])
                    print(max(params[metric]))
                    if score[metric] >= max(params[metric]):
                        with codecs.open(params['log_path']+'best_'+metric+'_prediction.txt','w','utf-8') as f:
                            f.write(codecs.open(params['log_path']+'candidate.txt','r','utf-8').read())
                        # ckpt保存模型
                        save_model(params['log_path']+'best_'+metric+'_checkpoint.pt', model, optim, params['updates'])
                model.train()
                params['report_loss'], params['report_time'] = 0, time.time()
                params['report_correct'], params['report_total'] = 0, 0
            # 保存模型，save_interval为3000，每3000步保存模型
            if params['updates'] % config.save_interval == 0:
                save_model(params['log_path']+'checkpoint.pt', model, optim, params['updates'])
    # 每个epoch都会更新学习率
    optim.updateLearningRate(score=0, epoch=epoch)

def eval_model(model, data, params):
    """
    模型评测，训练时使用交叉熵损失，而验证时使用rouge，造成了不一致
    :param model: 模型
    :param data: 训练时为trainloader
    :param params:各种参数
    :return: rouge分数
    """
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    count, total_count = 0, len(data['validset'])
    validloader = data['validloader']
    tgt_vocab = data['tgt_vocab']

    for src, tgt, src_len, tgt_len, original_src, original_tgt in validloader:
        if config.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()
        with torch.no_grad():
            if config.beam_size > 1: # 10
                samples, alignment, weight = model.beam_sample(src, src_len, beam_size=config.beam_size, eval_=True)
            else:
                samples, alignment = model.sample(src, src_len)
        # 将结果转换为label并加入list
        candidate += ["".join(tgt_vocab.convertToLabels(s, utils.EOS)) for s in samples]
        source += ["".join(ori_src) for ori_src in original_src]
        reference += [" ".join(list(ori_tgt[0])) for ori_tgt in original_tgt]
        if alignment is not None:
            alignments += [align for align in alignment]

        count += len(original_src)
        utils.progress_bar(count, total_count)

    if config.unk and config.attention != 'None':
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == utils.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append("".join(cand))
            if len(cand) == 0:
                print('Error!')
        candidate = cands
    # 写入文件
    with codecs.open(params['log_path']+'candidate.txt','w+','utf-8') as f:
        for i in range(len(candidate)):
            f.write(" ".join(candidate[i])+'\n')
    # 写入文件
    with codecs.open(params['log_path']+'reference.txt','w+','utf-8') as f:
        for i in range(len(reference)):
            f.write("".join(reference[i])+'\n')
    score = {}
    # 使用rouge进行评测
    for metric in config.metrics:
        score[metric] = getattr(utils, metric)(reference, candidate, params['log_path'], params['log'], config)

    return score

def save_model(path, model, optim, updates):
    """
    模型以字典形式存储
    :param path: 存储路径
    :param model: 存储模型
    :param optim: 存储optim
    :param updates: 存储更新次数
    :return: 无
    """
    model_state_dict = model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)

def build_log():
    """
    作用：创建日志
    参数：无
    返回值：
    """
    # 如果日志文件不存在，则创建文件夹
    if not os.path.exists(config.logF):
        os.mkdir(config.logF)
    # 若log为空，获取当前时间作为log路径
    if opt.log == '':
        log_path = config.logF + str(int(time.time() * 1000)) + '/'
    else:
        log_path = config.logF + opt.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    print_log = utils.print_log(log_path + 'log.txt')
    return print_log, log_path

def showAttention(path, s, c, attentions, index):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    ax.set_xticklabels([''] + s, rotation=90)
    ax.set_yticklabels([''] + c)
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    plt.savefig(path + str(index) + '.jpg')

def main():
    # checkpoint
    if opt.restore:
        print('loading checkpoint...\n')
        checkpoints = torch.load(opt.restore, map_location = 'cuda:%d' % opt.gpus[0])
    else:
        checkpoints = None
    # 调用load_data()函数，返回训练和验证集数据，以及两个词表
    data = load_data()
    # 创建日志，返回print_log函数的引用，传入内容可以写入日志，以及日志路径
    print_log, log_path = build_log()
    # 创建模型
    model, optim, print_log = build_model(checkpoints, print_log)
    # scheduler-false
    if config.schedule:
        scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)
    params = {'updates': 0, 'report_loss': 0, 'report_total': 0,
              'report_correct': 0, 'report_time': time.time(),
              'log': print_log, 'log_path': log_path}
    # 默认为rouge，params[rouge]
    for metric in config.metrics:
        params[metric] = []
    # 默认为空，restore checkpoint
    if opt.restore:
        params['updates'] = checkpoints['updates']
    # 训练模式
    if opt.mode == "train":
        # 进行训练模型，默认20个epoch
        for i in range(1, config.epoch + 1):
            # 默认为false
            if config.schedule:
                scheduler.step()
                print("Decaying learning rate to %g" % scheduler.get_lr()[0])
            # 调用train_model函数
            train_model(model, data, optim, i, params)
        # 打印rouge
        for metric in config.metrics:
            print_log("Best %s score: %s\n" % (metric, max(params[metric])))
    else:
        score = eval_model(model, data, params)

if __name__ == '__main__':
    main()
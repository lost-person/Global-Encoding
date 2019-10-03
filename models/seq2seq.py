import torch
import torch.nn as nn
import utils
import models
import random


class seq2seq(nn.Module):
    """
        seq2seq模型定义，encoder和decoder在rnn.py
    """
    def __init__(self, config, use_attention=True, encoder=None, decoder=None):
        super(seq2seq, self).__init__()
        # 如果encoder为None，则创建rnn_encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = models.rnn_encoder(config)
        # 如果是共享词表，则tgt_emb等于encoder的embedding
        tgt_embedding = self.encoder.embedding if config.shared_vocab else None
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = models.rnn_decoder(config, embedding=tgt_embedding, use_attention=use_attention)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.use_cuda = config.use_cuda
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=utils.PAD, reduction='none')
        if config.use_cuda:
            self.criterion.cuda()

    def compute_loss(self, scores, targets):
        """
        计算损失
        :param scores: 预测的概率分布
        :param targets: summary标签
        :return: 返回损失值
        """
        # scores [25, 64, 3363]
        scores = scores.view(-1, scores.size(2)) #[1600, 3363])
        loss = self.criterion(scores, targets.contiguous().view(-1))# [1600]
        return loss

    def forward(self, src, src_len, dec, targets, srcoov_pad=None, tgtoov_pad=None, oov_list=None, teacher_ratio=1.0):
        """
        :param src: 源文本，id表示，并padding，batch*max_len
        :param src_len: 文本长度
        :param dec: tgt文本id表示，无最后一个元素，第一个元素为<s>
        :param targets: tgt文本id表示，无第一个元素，最后一个元素为</s>
        :param teacher_ratio: 为默认值
        :return: 损失以及预测概率分布
        """
        src = src.t() # .t()表示转置
        dec = dec.t()
        targets = targets.t()
        teacher = random.random() < teacher_ratio # random()生成0-1之间的随机浮点数
        # 调用encoder，返回输出以及hn、cn
        contexts, state = self.encoder(src, src_len.tolist())

        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)
        outputs = []
        if self.config.pointer:
            max_oov = max([len(v) for v in oov_list])
            print(max_oov)
            if max_oov > 0:
                extra_zeros = torch.zeros(self.config.batch_size, max_oov)

        if teacher:
            for input in dec.split(1):# 按列切input_size=[1,64]，获取一个词
                output, state, attn_weights, p = self.decoder(input.squeeze(0), state) # squeeze去除维度=1的维度，将该词和state输入decoder
                outputs.append(output) # 保留所有预测的概率分布
            if self.config.pointer:
                output = torch.cat([output, extra_zeros], 1)
                print(output.size())
            outputs = torch.stack(outputs)
        else:
            inputs = [dec.split(1)[0].squeeze(0)] # 按列切input_size=[1,64]，取第一个组块，squeeze后大小[64]
            for i, _ in enumerate(dec.split(1)):
                output, state, attn_weights, p = self.decoder(inputs[i], state)

                # output为词表打分[batch, voc_size]
                predicted = output.max(1)[1]# 返回最大值在词表中的索引，作为下一个词的预测
                inputs += [predicted] # 保存所有预测的词
                outputs.append(output) # 保留所有预测概率分布
            outputs = torch.stack(outputs) # 默认为第一个维度
        # 计算损失
        loss = self.compute_loss(outputs, targets)
        return loss, outputs

    def sample(self, src, src_len):

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, reverse_indices = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        bos = torch.ones(src.size(0)).long().fill_(utils.BOS)
        src = src.t()

        if self.use_cuda:
            bos = bos.cuda()

        contexts, state = self.encoder(src, lengths.tolist())

        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)
        inputs, outputs, attn_matrix = [bos], [], []
        for i in range(self.config.max_time_step):
            output, state, attn_weights = self.decoder(inputs[i], state)
            predicted = output.max(1)[1]
            inputs += [predicted]
            outputs += [predicted]
            attn_matrix += [attn_weights]

        outputs = torch.stack(outputs)
        sample_ids = torch.index_select(outputs, dim=1, index=reverse_indices).t().tolist()

        if self.decoder.attention is not None:
            attn_matrix = torch.stack(attn_matrix)
            alignments = attn_matrix.max(2)[1]
            alignments = torch.index_select(alignments, dim=1, index=reverse_indices).t().tolist()
        else:
            alignments = None

        return sample_ids, alignments

    def beam_sample(self, src, src_len, beam_size=1, eval_=False):
        """
        谢谢小可爱！
        评测时生成摘要，使用beam_search，
        """
        # (1) Run the encoder on the src.
        # sort 返回排序结果和排序结果在原先 tensor 的索引
        print("小可爱棒棒哒！")
        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        # 返回原先 tensor 在排序后的索引
        _, ind = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        src = src.t()
        batch_size = src.size(1)
        contexts, encState = self.encoder(src, lengths.tolist())

        #  (1b) Initialize for the decoder.
        # 复制数据，且无需求导
        def var(a):
            return torch.tensor(a, requires_grad=False)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)
        # Repeat everything beam_size times.
        # contexts = rvar(contexts.data)
        contexts = rvar(contexts)

        if self.config.cell == 'lstm':
            decState = (rvar(encState[0]), rvar(encState[1]))
        else:
            decState = rvar(encState)

        # 自定义 Beam, length_norm 数据归一化
        beam = [models.Beam(beam_size, n_best=1,
                          cuda=self.use_cuda, length_norm=self.config.length_norm)
                for __ in range(batch_size)]
        if self.decoder.attention is not None:
            self.decoder.attention.init_context(contexts)

        # (2) run the decoder to generate sentences, using beam search.
        for i in range(self.config.max_time_step):
            if all((b.done() for b in beam)):
                break
            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(-1))
            # Run one step.
            output, decState, attn = self.decoder(inp, decState)
            # decOut: beam x rnn_size
            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
            # beam x tgt_vocab
            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output[:, j], attn[:, j])
                if self.config.cell == 'lstm':
                    b.beam_update(decState, j)
                else:
                    b.beam_update_gru(decState, j)

        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []
        if eval_:
            allWeight = []

        # for j in ind.data:
        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            if eval_:
                weight = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
                if eval_:
                    weight.append(att)
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])
            if eval_:
                allWeight.append(weight[0])
        
        if eval_:
            return allHyps, allAttn, allWeight

        return allHyps, allAttn

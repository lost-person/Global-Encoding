import argparse
import utils
import pickle

parser = argparse.ArgumentParser(description='preprocess.py')
parser.add_argument('-load_data', required=True,
                    help="input file for the data")
parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")
parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_filter', type=int, default=0,
                    help="Maximum source sequence length")
parser.add_argument('-tgt_filter', type=int, default=0,
                    help="Maximum target sequence length")
parser.add_argument('-src_trun', type=int, default=0,
                    help="Truncate source sequence length")
parser.add_argument('-tgt_trun', type=int, default=0,
                    help="Truncate target sequence length")
parser.add_argument('-src_char', action='store_true', help='character based encoding')
parser.add_argument('-tgt_char', action='store_true', help='character based decoding')
parser.add_argument('-src_suf', default='src',
                    help="the suffix of the source filename")
parser.add_argument('-tgt_suf', default='tgt', help="the suffix of the target filename")
# store_true就代表着一旦有这个参数，做出动作“将其值标为True”，也就是没有时，默认状态下其值为False
parser.add_argument('-share', action='store_true', help='share the vocabulary between source and target')
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")
parser.add_argument('-pointer', action='store_true', help='recording the oovs')
opt = parser.parse_args()

def makeVocabulary(filename, trun_length, filter_length, char, vocab, size):
    """
        作用：制作词表
        参数：文件名、截断长度(0)、长度限制(0)、opt.src_char/tgt_char（目标/源词表）, 传入初始词表（dicts['src']）, 词表大小
        返回值：词表
    """
    print("%s: length limit = %d, truncate length = %d" % (filename, filter_length, trun_length))
    max_length = 0
    with open(filename, encoding='utf8') as f:
        # 按行读入文件
        for sent in f.readlines():
            if char: # 若为True，将句子转换为列表（按字存储）
                tokens = list(sent.strip())
            else: # 按照空格分隔
                tokens = sent.strip().split()
            if 0 < filter_length < len(sent.strip().split()):
                continue
            # 获取最大句子长度
            max_length = max(max_length, len(tokens))
            # 根据截断长度进行截断
            if trun_length > 0:
                tokens = tokens[:trun_length]
            for word in tokens:
                # 已存在元素，不会进行添加
                vocab.add(word)

    print('Max length of %s = %d' % (filename, max_length))
    if size > 0:
        originalSize = vocab.size()
        vocab = vocab.prune(size)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))
    return vocab

def saveVocabulary(name, vocab, file):
    """
    作用: 保存词表（词表写入文件）
    参数：名称（src/tgt）、词表、保存文件名
    返回值: 无
    """
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)

def makeData(srcFile, tgtFile, srcDicts, tgtDicts, save_srcFile, save_tgtFile, lim=0):
    """
    作用: 建立词表与id之间的映射
    参数：源文件、目标文件、src词表、tgt词表、保存的src、保存的tgt
    返回值: 文件名及文件大小
    """
    sizes = 0
    count, empty_ignored, limit_ignored = 0, 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf8')
    tgtF = open(tgtFile, encoding='utf8')
    # 分别保存id和str
    srcIdF = open(save_srcFile + '.id', 'w')
    tgtIdF = open(save_tgtFile + '.id', 'w')
    srcStrF = open(save_srcFile + '.str', 'w', encoding='utf8')
    tgtStrF = open(save_tgtFile + '.str', 'w', encoding='utf8')
    # ---- 修改
    if opt.pointer:
        srcIdOOvF = open(save_srcFile + '_oov.id', 'w', encoding='utf8')
        tgtIdOOvF = open(save_tgtFile + '_oov.id', 'w', encoding='utf8')
        OOvF = open(save_srcFile + '_oov.str', 'w', encoding='utf8')
    # ----
    while True:
        sline = srcF.readline()
        tline = tgtF.readline()
        # 到文件末尾
        if sline == "" and tline == "":
            break
        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break
        # 去除首尾空格
        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            empty_ignored += 1
            continue

        sline = sline.lower()
        tline = tline.lower()
        # 若char为true，则为中文格式使用list，否则为英文格式
        srcWords = sline.split() if not opt.src_char else list(sline)
        tgtWords = tline.split() if not opt.tgt_char else list(tline)

        # 若未超出句子长度限制
        if (opt.src_filter == 0 or len(sline.split()) <= opt.src_filter) and \
           (opt.tgt_filter == 0 or len(tline.split()) <= opt.tgt_filter):
            # 若对句子长度进行限制，则截断
            if opt.src_trun > 0:
                srcWords = srcWords[:opt.src_trun]
            if opt.tgt_trun > 0:
                tgtWords = tgtWords[:opt.tgt_trun]
            srcIds = srcDicts.convertToIdx(srcWords, utils.UNK_WORD)
            # 目标ids中加入句子开始与结束，普通词表
            tgtIds = tgtDicts.convertToIdx(tgtWords, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD)
            if opt.pointer: # 若有指针机制，需要oov列表进行词表扩展
                srcIdOOvs, oovs = srcDicts.convertToIdxandOOVs(srcWords, utils.UNK_WORD)
                tgtIdOOvs = tgtDicts.convertToIdxwithOOVs(tgtWords, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD, oovs)
                srcIdOOvF.write(" ".join(list(map(str, srcIdOOvs)))+'\n')
                tgtIdOOvF.write(" ".join(list(map(str, tgtIdOOvs)))+'\n')

            # 句子映射id写入
            srcIdF.write(" ".join(list(map(str, srcIds)))+'\n')
            tgtIdF.write(" ".join(list(map(str, tgtIds)))+'\n')

            if not opt.src_char:
                srcStrF.write(" ".join(srcWords)+'\n')
                if opt.pointer:
                    OOvF.write(" ".join(oovs) + '\n')
            else:
                srcStrF.write("".join(srcWords) + '\n')
                if opt.pointer:
                    OOvF.write("".join(oovs) + '\n')
            if not opt.tgt_char:
                tgtStrF.write(" ".join(tgtWords)+'\n')
            else:
                tgtStrF.write("".join(tgtWords) + '\n')

            sizes += 1
        else:
            limit_ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()
    srcStrF.close()
    tgtStrF.close()
    srcIdF.close()
    tgtIdF.close()

    print('Prepared %d sentences (%d and %d ignored due to length == 0 or > )' %
          (sizes, empty_ignored, limit_ignored))
    if opt.pointer:
        srcIdOOvF.close()
        tgtIdOOvF.close()
        OOvF.close()
        return {'srcF': save_srcFile + '.id', 'tgtF': save_tgtFile + '.id',
            'original_srcF': save_srcFile + '.str', 'original_tgtF': save_tgtFile + '.str',
            'length': sizes, 'srcoovF': save_srcFile + '_oov.id',  'tgtoovF': save_tgtFile + '_oov.id', 'oov': oovs, 'oovF': save_srcFile + '_oov.str'}
    return {'srcF': save_srcFile + '.id', 'tgtF': save_tgtFile + '.id',
            'original_srcF': save_srcFile + '.str', 'original_tgtF': save_tgtFile + '.str',
            'length': sizes}
def main():
    dicts = {}
    # load_data中存放输入文件夹路径、分别需要使用6个文件（train.src、train.tgt。。。）
    train_src, train_tgt = opt.load_data + 'train.' + opt.src_suf, opt.load_data + 'train.' + opt.tgt_suf
    valid_src, valid_tgt = opt.load_data + 'valid.' + opt.src_suf, opt.load_data + 'valid.' + opt.tgt_suf
    test_src, test_tgt = opt.load_data + 'test.' + opt.src_suf, opt.load_data + 'test.' + opt.tgt_suf

    # 保存文件名及路径
    save_train_src, save_train_tgt = opt.save_data + 'train.' + opt.src_suf, opt.save_data + 'train.' + opt.tgt_suf
    save_valid_src, save_valid_tgt = opt.save_data + 'valid.' + opt.src_suf, opt.save_data + 'valid.' + opt.tgt_suf
    save_test_src, save_test_tgt = opt.save_data + 'test.' + opt.src_suf, opt.save_data + 'test.' + opt.tgt_suf

    # 词表保存路径
    src_dict, tgt_dict = opt.save_data + 'src.dict', opt.save_data + 'tgt.dict'

    # 判断输入输出是否共享词表， 默认为false
    if opt.share:
        assert opt.src_vocab_size == opt.tgt_vocab_size
        print('Building source and target vocabulary...')
        # 创建词表，源和目标词表相同。调用utils中的dict_helper.py中的Dict类，[<blank>,<unk>,<s>,</s>]。
        dicts['src'] = dicts['tgt'] = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
        # 调用makeVocabulary构建词表
        dicts['src'] = makeVocabulary(train_src, opt.src_trun, opt.src_filter, opt.src_char, dicts['src'], opt.src_vocab_size)
        # 将target也放入词表
        dicts['src'] = dicts['tgt'] = makeVocabulary(train_tgt, opt.tgt_trun, opt.tgt_filter, opt.tgt_char, dicts['tgt'], opt.tgt_vocab_size)
    else:
        print('Building source vocabulary...')
        dicts['src'] = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
        dicts['src'] = makeVocabulary(train_src, opt.src_trun, opt.src_filter, opt.src_char, dicts['src'], opt.src_vocab_size)
        print('Building target vocabulary...')
        dicts['tgt'] = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
        dicts['tgt'] = makeVocabulary(train_tgt, opt.tgt_trun, opt.tgt_filter, opt.tgt_char, dicts['tgt'], opt.tgt_vocab_size)

    print('Preparing training ...')
    train = makeData(train_src, train_tgt, dicts['src'], dicts['tgt'], save_train_src, save_train_tgt)

    print('Preparing validation ...')
    valid = makeData(valid_src, valid_tgt, dicts['src'], dicts['tgt'], save_valid_src, save_valid_tgt)

    print('Preparing test ...')
    test = makeData(test_src, test_tgt, dicts['src'], dicts['tgt'], save_test_src, save_test_tgt)

    print('Saving source vocabulary to \'' + src_dict + '\'...')
    dicts['src'].writeFile(src_dict)

    print('Saving source vocabulary to \'' + tgt_dict + '\'...')
    dicts['tgt'].writeFile(tgt_dict)

    data = {'train': train, 'valid': valid,
             'test': test, 'dict': dicts}
    pickle.dump(data, open(opt.save_data+'data.pkl', 'wb'))

if __name__ == "__main__":
    main()
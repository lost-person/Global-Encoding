# Global-Encoding
This is the code for our paper *Global Encoding for Abstractive Summarization*, https://arxiv.org/abs/1805.03989

***********************************************************

## Requirements
* Ubuntu 16.0.4
* Python 3.5
* Pytorch 0.4.1 (updated)
* pyrouge

In order to use pyrouge, set rouge path with the line below:
```
pyrouge_set_rouge_path RELEASE-1.5.5/
```
It seems that some user have met problems with pyrouge, so I have updated the script, and users can put the directory "RELEASE-1.5.5" in your home directory and set rouge path to it  (or run the command "chmod 777 RELEASE-1.5.5" for the permission).
**************************************************************

## Preprocessing
```
python3 preprocess.py -load_data path_to_data -save_data path_to_store_data 
```
Remember to put the data into a folder and name them *train.src*, *train.tgt*, *valid.src*, *valid.tgt*, *test.src* and *test.tgt*, and make a new folder inside called *data*

***************************************************************

## Training
```
python3 train.py -log log_name -config config_yaml -gpus id
```
Create your own yaml file for hyperparameter setting.

****************************************************************

## Evaluation
```
python3 train.py -log log_name -config config_yaml -gpus id -restore checkpoint -mode eval
```

*******************************************************************

# Citation
If you use this code for your research, please kindly cite our paper:
```
@inproceedings{globalencoding,
  title     = {Global Encoding for Abstractive Summarization},
  author    = {Junyang Lin and Xu Sun and Shuming Ma and Qi Su},
  booktitle = {{ACL} 2018},
  year      = {2018}
}
```

改进：
1. 增加指针网络；—— 首先构建原始词表，然后在将文章和摘要转换为id时，对 oov 特殊处理，即将文章中的oov的单词额外记录，以便解码生成摘要时，进行copy；

2. 词表处理详见 dict_helper.py 的 convertToIdxandOOVs 函数 和 pointer_summarizer-master 的 data.py 的article2ids, abstrac2ids, outputids2words 函数。后者的项目中，在的article2ids中，额外记录oov单词，并用在abstract2ids，以实现将指针机制。最后在解码时，outputids2word利用idx进行解码。

3. 前两个函数，在 batcher.py 中的 example 的初始化时完成，后者在 decoder.py 的 decode 函数中完成。因此，我们可以依样画葫芦，在 preprocess.py 中 makeData 时完成 oov 的记录。
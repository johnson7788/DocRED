# Baseline code

## 0. Requirements and Installation
python3

pytorch>=1.0

```
pip3 install -r requirements.txt
```

## 1. 准备数据，根据原始数据，生成各种向量数据，保存到prepro_data

Download metadata from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/99e1c0805eb64736af95/) or [GoogleDrive](https://drive.google.com/drive/folders/1Ri3LIILKKBi3aBJjUVCOBpGX5PpONHRK) for baseline method and put them into prepro_data folder.
```buildoutcfg
确保DocRED文件下包括
data/
├── README.md
├── dev.json
├── rel_info.json
├── test.json
├── train_annotated.json
└── train_distant.json

确保prepro_data下存在 rel2id.json文件, 从google drive下载
code/prepro_data/
├── char2id.json  #字符到id的映射 "BLANK": 0, "UNK": 1, "t": 2,
├── ner2id.json  #实体类型到id的映射 {"BLANK": 0, "ORG": 1, "LOC": 2, "TIME": 3, "PER": 4, "MISC": 5, "NUM": 6}
├── rel2id.json  关系到id的映射，例如"P1376": 79, "P607": 27,
├── word2id.json  #单词到id的映射
├── char_vec.npy   #字符向量
├── vec.npy     #单词向量
```
### 开始生成训练需要的数据
```
python3 gen_data.py --in_path ../data --out_path prepro_data

开始处理文件../data/train_annotated.json
加载json文件，可能耗时较长
一共收集到数据3053条数据
原始的数据的个数: 3053
生成的数据的个数: 3053
开始生成id到向量的映射文件，保存成npy格式
保存数据文件: prepro_data/train_ano.json, 耗时可能较长
开始加载char2id.json,  word2id.json, ner2id.json
开始保存各种id映射文件
保存完成: 每个样本单词到id的映射文件prepro_data/train_ano_word.npy, 形状是:(3053, 512)
保存完成: 每个样本实体的位置到id的映射文件prepro_data/train_ano_pos.npy, 形状是:(3053, 512)
保存完成: 每个样本实体类型到id的映射文件prepro_data/train_ano_ner.npy, 形状是:(3053, 512)
保存完成: 每个样本字符到id的映射文件prepro_data/train_ano_char.npy, 形状是:(3053, 512, 16)

开始处理文件../data/train_distant.json
加载json文件，可能耗时较长
一共收集到数据101873条数据
原始的数据的个数: 101873
生成的数据的个数: 101873
开始生成id到向量的映射文件，保存成npy格式
保存数据文件: prepro_data/train_dis.json, 耗时可能较长
开始加载char2id.json,  word2id.json, ner2id.json
开始保存各种id映射文件
保存完成: 每个样本单词到id的映射文件prepro_data/train_dis_word.npy, 形状是:(101873, 512)
保存完成: 每个样本实体的位置到id的映射文件prepro_data/train_dis_pos.npy, 形状是:(101873, 512)
保存完成: 每个样本实体类型到id的映射文件prepro_data/train_dis_ner.npy, 形状是:(101873, 512)
保存完成: 每个样本字符到id的映射文件prepro_data/train_dis_char.npy, 形状是:(101873, 512, 16)

开始处理文件../data/dev.json
加载json文件，可能耗时较长
一共收集到数据1000条数据
原始的数据的个数: 1000
生成的数据的个数: 1000
开始生成id到向量的映射文件，保存成npy格式
保存数据文件: prepro_data/dev_dev.json, 耗时可能较长
开始加载char2id.json,  word2id.json, ner2id.json
开始保存各种id映射文件
保存完成: 每个样本单词到id的映射文件prepro_data/dev_dev_word.npy, 形状是:(1000, 512)
保存完成: 每个样本实体的位置到id的映射文件prepro_data/dev_dev_pos.npy, 形状是:(1000, 512)
保存完成: 每个样本实体类型到id的映射文件prepro_data/dev_dev_ner.npy, 形状是:(1000, 512)
保存完成: 每个样本字符到id的映射文件prepro_data/dev_dev_char.npy, 形状是:(1000, 512, 16)

开始处理文件../data/test.json
加载json文件，可能耗时较长
一共收集到数据1000条数据
原始的数据的个数: 1000
生成的数据的个数: 1000
开始生成id到向量的映射文件，保存成npy格式
保存数据文件: prepro_data/dev_test.json, 耗时可能较长
开始加载char2id.json,  word2id.json, ner2id.json
开始保存各种id映射文件
保存完成: 每个样本单词到id的映射文件prepro_data/dev_test_word.npy, 形状是:(1000, 512)
保存完成: 每个样本实体的位置到id的映射文件prepro_data/dev_test_pos.npy, 形状是:(1000, 512)
保存完成: 每个样本实体类型到id的映射文件prepro_data/dev_test_ner.npy, 形状是:(1000, 512)
保存完成: 每个样本字符到id的映射文件prepro_data/dev_test_char.npy, 形状是:(1000, 512, 16)

#生成的结果
code/prepro_data/
├── README.md
├── char2id.json
├── char_vec.npy
├── dev_dev.json
├── dev_dev_char.npy
├── dev_dev_ner.npy
├── dev_dev_pos.npy
├── dev_dev_word.npy
├── dev_test.json
├── dev_test_char.npy
├── dev_test_ner.npy
├── dev_test_pos.npy
├── dev_test_word.npy
├── id2rel.json
├── ner2id.json
├── rel2id.json
├── train_ano.json
├── train_ano_char.npy
├── train_ano_ner.npy
├── train_ano_pos.npy
├── train_ano_word.npy
├── train_dis.json
├── train_dis_char.npy
├── train_dis_ner.npy
├── train_dis_pos.npy
├── train_dis_word.npy
├── vec.npy
└── word2id.json
```

## 2. 任务1， 关系抽取
支持的模型包括, CNN3,LSTM,BiLSTM,ContextAware
training:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name BiLSTM --save_name anotation_BiLSTM --do_train --do_dev --do_test

#默认不是用GPU，如果使用，需要
python3 train.py --model_name BiLSTM --save_name anotation_BiLSTM --do_train --do_dev --do_test --gpu
```

testing (--test_prefix dev_dev for dev set, dev_test for test set):
```
CUDA_VISIBLE_DEVICES=0 python3 test.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev --input_theta 0.3601
```

## 3. 任务2， 支持证据的抽取, sp代表support evidence

training:
```
CUDA_VISIBLE_DEVICES=0 python3 train_sp.py --model_name LSTM_SP  --save_name checkpoint_BiLSTMSP  --train_prefix dev_train --test_prefix dev_dev
```

testing:
```
CUDA_VISIBLE_DEVICES=0 python3 test_sp.py --model_name LSTM_SP --save_name checkpoint_BiLSTMSP --train_prefix dev_train --test_prefix dev_dev --input_theta 0.4619
```

## 4. 评估

使用dev数据集评估
```
python3 evalutaion result.json ../data/dev.json
```





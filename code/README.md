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
├── char2id.json
├── ner2id.json
├── rel2id.json
├── word2id.json
├── char_vec.npy
├── vec.npy
```
### 开始生成训练需要的数据
```
python3 gen_data.py --in_path ../data --out_path prepro_data

#生成的结果
code/prepro_data/
├── README.md
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
├── dev_train.json
├── dev_train_char.npy
├── dev_train_ner.npy
├── dev_train_pos.npy
├── dev_train_word.npy
├── id2rel.json
├── train.json
├── train_char.npy
├── train_ner.npy
├── train_pos.npy
├── train_word.npy
```

## 2. 任务1， 关系抽取
支持的模型包括, CNN3,LSTM,BiLSTM,ContextAware
training:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev
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





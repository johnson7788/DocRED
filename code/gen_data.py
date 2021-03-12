import numpy as np
import os
import json
from nltk.tokenize import WordPunctTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default="../data", help='读取原数据，原数据是标注好的json文件')
parser.add_argument('--out_path', type=str, default="prepro_data", help='处理原数据，生成训练数据')

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False
#一个单词最多由多少个字符组成，这里是设置16个，嵌入矩阵的维度的最后一维是16
char_limit = 16
# 远端监督的训练数据和人工标注的训练数据，还有测试和开发数据
train_distant_file_name = os.path.join(in_path, 'train_distant.json')
train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

# 关系到id的映射
rel2id = json.load(open(os.path.join(out_path, 'rel2id.json'), "r"))
id2rel = {v: u for u, v in rel2id.items()}
# 生成id到关系的映射
json.dump(id2rel, open(os.path.join(out_path, 'id2rel.json'), "w"))
# 存储对实体名称和关系,只是训练的
fact_in_train = set([])
# 存储对实体名称和关系，是开发集的 eg: {('Asian Spirit and Zest Air', 'Pasay City', 'P159'), ('Zest Airways, Inc.', 'Pasay City', 'P159')}
fact_in_dev_train = set([])


def init(data_file_name, rel2id, max_length=512, is_training=True, suffix=''):
    """
    :param data_file_name: 要处理的json文件
    :param rel2id: 关系到id的映射
    :param max_length: 整个文档的最大序列长度，即拼接所有句子，最大长度512是否合理？
    :param is_training: 是否是训练，如果是，生成train_suffix.json的文件
    :param suffix: 文件的命名的后半部分
    :return:
    """
    print(f"开始处理文件{data_file_name}")
    print(f"加载json文件，可能耗时较长")
    ori_data = json.load(open(data_file_name))
    print(f"一共收集到数据{len(ori_data)}条数据")
    #所有数据中实体数量最多的值
    Ma = 0
    #所有数据中关系最多的值
    Ma_e = 0
    #存储处理后的每条数据
    data = []
    intrain = notintrain = notindevtrain = indevtrain = 0
    for i in range(len(ori_data)):
        # 用来记录，每个句子的分隔点，例如 eg[0, 32, 43, 97, 125, 175]， 0到32是第一个句子的长度，里面包含32个单词
        Ls = [0]
        # 所有句子中的单词总数
        L = 0
        # 循环每个句子，ori_data的一条数据中有很多句子
        for x in ori_data[i]['sents']:
            L += len(x)
            Ls.append(L)
        #开始处理实体
        vertexSet = ori_data[i]['vertexSet']
        # 把实体在单个句子中的位置，转换成在整个文档中的位置
        for j in range(len(vertexSet)):
            #循环每种实体，例如一个人的名字可能有多种叫法'Miguel Riofrio Sánchez'，'Riofrio'，'Miguel Riofrío'
            for k in range(len(vertexSet[j])):
                #循环每个实体，在哪个句子中的id转换成int类型
                vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])
                # 获取下sent_id
                sent_id = vertexSet[j][k]['sent_id']
                # 根据句子id，句子的的分隔点
                dl = Ls[sent_id]
                #获取实体的在单个句子中的位置
                pos1 = vertexSet[j][k]['pos'][0]
                pos2 = vertexSet[j][k]['pos'][1]
                #把实体在单个句子中的位置，转换成在整个文档中的位置
                vertexSet[j][k]['pos'] = (pos1 + dl, pos2 + dl)

        ori_data[i]['vertexSet'] = vertexSet
        #存储一条修改后的数据
        item = {}
        item['vertexSet'] = vertexSet
        #获取一下labels, 如果是train，才会有
        labels = ori_data[i].get('labels', [])
        # 存储 vertexSet中的第一个实体的索引, label['t']第二个实体的索引
        train_triple = set([])
        new_labels = []
        #处理每个label
        for label in labels:
            # eg: 'P35'
            rel = label['r']
            # 确保label在关系到id的表中
            assert (rel in rel2id)
            # 把 关系转换成id, 'P35'-->54
            label['r'] = rel2id[rel]
            # label['h'] vertexSet中的第一个实体的索引, label['t']第二个实体的索引
            train_triple.add((label['h'], label['t']))
            #如果后缀是_train
            if suffix == '_train':
                #循环所有实体的名称，一个实体可能有不同名称，但是他们之间的关系是相同的
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_dev_train.add((n1['name'], n2['name'], rel))

            if is_training:
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_train.add((n1['name'], n2['name'], rel))

            else:
                #加2个参数？
                label['intrain'] = False
                label['indev_train'] = False

                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        if (n1['name'], n2['name'], rel) in fact_in_train:
                            label['intrain'] = True

                        if suffix == '_dev' or suffix == '_test':
                            if (n1['name'], n2['name'], rel) in fact_in_dev_train:
                                label['indev_train'] = True
            #修改后的加到新的labels列表中
            new_labels.append(label)
        #存储labels和title
        item['labels'] = new_labels
        item['title'] = ori_data[i]['title']
        #没有关系的实体，好像有些问题？我们标注的第一个实体和第二个实体有关系，假设是（1，2）实体对有关系，然后实体对（2，1）就没有关系了吗
        na_triple = []
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet)):
                if (j != k):
                    if (j, k) not in train_triple:
                        na_triple.append((j, k))
        # 没有关系的实体对
        item['na_triple'] = na_triple
        # 每个句子的分隔点
        item['Ls'] = Ls
        item['sents'] = ori_data[i]['sents']
        #存储到总的数据中
        data.append(item)
        # 获取实体数量和关系数量的最大值
        Ma = max(Ma, len(vertexSet))
        Ma_e = max(Ma_e, len(item['labels']))

    print('原始的数据的个数:', len(ori_data))
    print('生成的数据的个数:', len(data))
    # print ('Ma_V', Ma)
    # print ('Ma_e', Ma_e)
    # print (suffix)
    # print ('fact_in_train', len(fact_in_train))
    # print (intrain, notintrain)
    # print ('fact_in_devtrain', len(fact_in_dev_train))
    # print (indevtrain, notindevtrain)

    # saving
    print("开始生成id到向量的映射文件，保存成npy格式")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"
    #数据保存
    dfile = os.path.join(out_path, name_prefix + suffix + '.json')
    print(f"保存数据文件: {dfile}, 耗时可能较长")
    json.dump(data, open(dfile, "w"))
    print(f"开始加载char2id.json,  word2id.json, ner2id.json")
    char2id = json.load(open(os.path.join(out_path, "char2id.json")))
    # id2char= {v:k for k,v in char2id.items()}
    # json.dump(id2char, open("data/id2char.json", "w"))

    word2id = json.load(open(os.path.join(out_path, "word2id.json")))
    ner2id = json.load(open(os.path.join(out_path, "ner2id.json")))
    #获取数据的个数
    sen_tot = len(ori_data)
    #初始化全0矩阵，[数据个数，最大序列长度],训练集的 (3053, 512), 存储所有句子和单词等的数字映射
    sen_word = np.zeros((sen_tot, max_length), dtype=np.int64)
    # eg: 训练集的 (3053, 512)， 存在实体的位置标为1，不存在的为0
    sen_pos = np.zeros((sen_tot, max_length), dtype=np.int64)
    # eg: 训练集的 (3053, 512)， 实体的类型，在实体的位置处设置实体的类型的id
    sen_ner = np.zeros((sen_tot, max_length), dtype=np.int64)
    # eg: 训练集的 (3053, 512, 16)
    sen_char = np.zeros((sen_tot, max_length, char_limit), dtype=np.int64)

    for i in range(len(ori_data)):
        item = ori_data[i]
        # 把一个句子的所有单词拼接到一起
        words = []
        for sent in item['sents']:
            words += sent

        for j, word in enumerate(words):
            # 单词都转换成小写
            word = word.lower()
            # 在最大长度以内的单词， 超过最大长度的，丢弃
            if j < max_length:
                #转换成id或UNK
                if word in word2id:
                    sen_word[i][j] = word2id[word]
                else:
                    sen_word[i][j] = word2id['UNK']
                #对于一个字，其中的字符也映射为id
                for c_idx, k in enumerate(list(word)):
                    if c_idx >= char_limit:
                        break
                    sen_char[i, j, c_idx] = char2id.get(k, char2id['UNK'])
        #对于少于最大序列长度的那些文档，填充BLANK
        for j in range(j + 1, max_length):
            sen_word[i][j] = word2id['BLANK']

        vertexSet = item['vertexSet']

        for idx, vertex in enumerate(vertexSet, 1):
            for v in vertex:
                #循环每一个实体, 存在实体的，那么标为1，不存在的为0
                sen_pos[i][v['pos'][0]:v['pos'][1]] = idx
                #实体的类型，在实体的位置处设置实体的类型的id
                sen_ner[i][v['pos'][0]:v['pos'][1]] = ner2id[v['type']]

    print("开始保存各种向量文件")
    word_id_file = os.path.join(out_path, name_prefix + suffix + '_word.npy')
    pos_id_file = os.path.join(out_path, name_prefix + suffix + '_pos.npy')
    ner_id_file = os.path.join(out_path, name_prefix + suffix + '_ner.npy')
    char_id_file = os.path.join(out_path, name_prefix + suffix + '_char.npy')
    np.save(word_id_file, sen_word)
    np.save(pos_id_file, sen_pos)
    np.save(ner_id_file, sen_ner)
    np.save(char_id_file, sen_char)
    print(f"保存完成: 每个样本单词到id的映射文件:{word_id_file}, 形状是:{sen_word.shape}")
    print(f"保存完成: 每个样本实体的位置到id的映射文件:{pos_id_file}, 形状是:{sen_pos.shape}")
    print(f"保存完成: 每个样本实体类型到id的映射文件:{ner_id_file}, 形状是:{sen_ner.shape}")
    print(f"保存完成: 每个样本字符到id的映射文件:{char_id_file}, 形状是:{sen_char.shape}")
    print()


if __name__ == '__main__':
    # 逐个处理所有的文件, 如果训练标注的数据，就让is_training为True，如果训练远端监督数据，就让远端的为True
    init(train_annotated_file_name, rel2id, max_length=512, is_training=True, suffix='_ano')
    init(train_distant_file_name, rel2id, max_length=512, is_training=True, suffix='_dis')
    init(dev_file_name, rel2id, max_length=512, is_training=False, suffix='_dev')
    init(test_file_name, rel2id, max_length=512, is_training=False, suffix='_test')


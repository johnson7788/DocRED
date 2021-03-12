# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
from tqdm.auto import trange
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import torch.nn.functional as F

IGNORE_INDEX = -100
is_transformer = False


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


class Config(object):
    def __init__(self, args):
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.data_path = './prepro_data/'
        self.use_bag = False
        # 是否使用GPU
        self.use_gpu = args.gpu
        self.is_training = True
        self.max_length = 512
        self.pos_num = 2 * self.max_length
        self.entity_num = self.max_length
        self.relation_num = 97
        #??这是什么
        self.coref_size = 20
        self.entity_type_size = 20
        # 训练多少epoch
        self.max_epoch = args.epoch
        self.opt_method = 'Adam'
        self.optimizer = None

        self.checkpoint_dir = './checkpoint'
        self.fig_result_dir = './fig_result'
        self.test_epoch = 5
        # 是否使用预训练的模型参数,如果使用，在模型初始化后，会自动加载self.pretrain_model
        self.pretrain_model = None
        self.word_size = 100
        self.epoch_range = None
        self.cnn_drop_prob = 0.5  # for cnn
        self.keep_prob = 0.8  # for lstm
        # 每多少个step，打印一次日志
        self.period = 1
        self.batch_size = 8
        #假设一篇文档中实体之间的关系最多为1800个，其实没那么多，我觉得可以少一点，过滤出实体最多的那篇文档，设置一下即可
        self.h_t_limit = 1800
        self.test_batch_size = self.batch_size
        self.test_relation_limit = 1800
        self.char_limit = 16
        self.sent_limit = 25
        self.dis2idx = np.zeros((512), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20
        # 单词向量
        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
        # 字符向量
        self.data_char_vec = np.load(os.path.join(self.data_path, 'char_vec.npy'))
        # label映射为id
        self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        if not os.path.exists("log"):
            os.mkdir("log")

    def load_train_data(self, train_word_file, train_pos_file, train_ner_file, train_char_file, train_json_file):
        print("开始读取训练数据")
        print(f"开始加载训练的词向量{train_word_file}, 实体位置向量{train_pos_file}, 实体类型向量{train_ner_file}，字符向量{train_char_file},训练json文件{train_json_file}")
        self.data_train_word = np.load(train_word_file)
        self.data_train_pos = np.load(train_pos_file)
        self.data_train_ner = np.load(train_ner_file)
        self.data_train_char = np.load(train_char_file)
        self.train_file = json.load(open(train_json_file))
        print("加载完成")
        #ins_num 是样本个数， self.train_len也是获取样本个数
        self.train_len = ins_num = self.data_train_word.shape[0]
        assert (self.train_len == len(self.train_file))
        self.train_order = list(range(ins_num))
        #计算一个epoch有多少batch
        self.train_batches = ins_num // self.batch_size
        if ins_num % self.batch_size != 0:
            self.train_batches += 1

    def load_test_data(self, test_word_file, test_pos_file, test_ner_file, test_char_file, test_file, mode='test'):
        print("开始读取测试数据")
        self.is_test = ('test' == mode)
        self.data_test_word = np.load(test_word_file)
        self.data_test_pos = np.load(test_pos_file)
        self.data_test_ner = np.load(test_ner_file)
        self.data_test_char = np.load(test_char_file)
        self.test_file = json.load(open(test_file))

        self.test_len = self.data_test_word.shape[0]
        assert (self.test_len == len(self.test_file))

        print("Finish reading")

        self.test_batches = self.data_test_word.shape[0] // self.test_batch_size
        if self.data_test_word.shape[0] % self.test_batch_size != 0:
            self.test_batches += 1

        self.test_order = list(range(self.test_len))
        self.test_order.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)

    def get_train_batch(self):
        random.shuffle(self.train_order)
        if self.use_gpu:
            context_idxs = torch.LongTensor(self.batch_size, self.max_length).cuda()
            context_pos = torch.LongTensor(self.batch_size, self.max_length).cuda()
            h_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
            t_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
            relation_multi_label = torch.Tensor(self.batch_size, self.h_t_limit, self.relation_num).cuda()
            relation_mask = torch.Tensor(self.batch_size, self.h_t_limit).cuda()

            pos_idx = torch.LongTensor(self.batch_size, self.max_length).cuda()

            context_ner = torch.LongTensor(self.batch_size, self.max_length).cuda()
            context_char_idxs = torch.LongTensor(self.batch_size, self.max_length, self.char_limit).cuda()

            relation_label = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()

            ht_pair_pos = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()
        else:
            #随机初始化特征
            #单词id初始化
            context_idxs = torch.LongTensor(self.batch_size, self.max_length)
            # 上下文位置初始化
            context_pos = torch.LongTensor(self.batch_size, self.max_length)
            # h_mapping表示第一个实体的id
            h_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length)
            # t_mapping表示第二个实体的id
            t_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length)
            # 关系标签, 是要预测的标签，多标签预测，第i个句子的第j个实体的关系是r
            relation_multi_label = torch.Tensor(self.batch_size, self.h_t_limit, self.relation_num)
            relation_mask = torch.Tensor(self.batch_size, self.h_t_limit)
            # 实体位置初始化
            pos_idx = torch.LongTensor(self.batch_size, self.max_length)
            #实体初始化
            context_ner = torch.LongTensor(self.batch_size, self.max_length)
            context_char_idxs = torch.LongTensor(self.batch_size, self.max_length, self.char_limit)

            relation_label = torch.LongTensor(self.batch_size, self.h_t_limit)
            # 2个实体首次出现的距离作为一个特征, shape torch.Size([8, 1800])
            ht_pair_pos = torch.LongTensor(self.batch_size, self.h_t_limit)

        for b in tqdm(range(self.train_batches), desc='Batch_Data: '):
            start_id = b * self.batch_size
            cur_bsz = min(self.batch_size, self.train_len - start_id)
            #cur_batch， 一个batch中包含的数据的索引位置
            cur_batch = list(self.train_order[start_id: start_id + cur_bsz])
            cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x] > 0), reverse=True)
            # 把h_mapping和t_mapping中所有值都设为0
            for mapping in [h_mapping, t_mapping]:
                mapping.zero_()
            # 把relation_multi_label和relation_mask, pos_idx中所有值都设为0, 要预测出来的标签
            for mapping in [relation_multi_label, relation_mask, pos_idx]:
                mapping.zero_()
            # ht_pair_pos都设为0
            ht_pair_pos.zero_()
            #全部用-100填充，默认填充-100，表示所有位置都没有关系
            relation_label.fill_(IGNORE_INDEX)
            #有关系样本+没有关系样本的最大的数量，
            max_h_t_cnt = 1
            #开始迭代一个batch的数据
            for i, index in enumerate(tqdm(cur_batch, desc='Current Batch: ', disable=False)):
                #从预处理数据中取出训练数据
                context_idxs[i].copy_(torch.from_numpy(self.data_train_word[index, :]))
                context_pos[i].copy_(torch.from_numpy(self.data_train_pos[index, :]))
                context_char_idxs[i].copy_(torch.from_numpy(self.data_train_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_train_ner[index, :]))
                # pos_idx是一个[1,2,3,4,......]的tensor，表示的是这一条数据这个文档的这个单词位置是否有单词，相当于确定文档的长度
                for j in range(self.max_length):
                    if self.data_train_word[index, j] == 0:
                        break
                    pos_idx[i, j] = j + 1
                #获取样本的全部信息，获取相比的标签
                ins = self.train_file[index]
                labels = ins['labels']
                idx2label = defaultdict(list)
                # 生成类似， 实体的在vertex的位置id和label的id的映射表。defaultdict(<class 'list'>, {(2, 3): [4], (2, 7): [4], (3, 2): [2, 1], (3, 0): [42], (18, 2): [3, 14], (20, 2): [1], (20, 3): [24], (7, 2): [2, 1], (11, 2): [3], (19, 2): [1], (0, 1): [7], (0, 2): [3], (0, 3): [69]})
                for label in labels:
                    idx2label[(label['h'], label['t'])].append(label['r'])
                #train_tripe 所有要训练的实体对在vertexSet中的位置
                train_tripe = list(idx2label.keys())
                for j, (h_idx, t_idx) in enumerate(train_tripe):
                    # 实体信息hlist和tlist
                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]
                    for h in hlist:
                        # i表示第i个样本，j表示第j个实体，最后一维表示位置信息，这个位置的id和这个实体出现的次数和位置有关,对应论文中的mk的定义
                        h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                    for t in tlist:
                        t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])
                    #获取label
                    label = idx2label[(h_idx, t_idx)]
                    # 对应论文中 dij和dji为文档中两个实体首次提及的相对距离
                    delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
                    if delta_dis < 0:
                        ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                    else:
                        ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
                    #relation_multi_label 是要预测的标签，多标签预测，第i个句子的第j个实体的关系是r
                    for r in label:
                        relation_multi_label[i, j, r] = 1
                    # relation_mask shape, [batch_size, max_num_releations], eg: torch.Size([8, 1800])
                    relation_mask[i, j] = 1
                    #随机填充一个假的关系
                    rt = np.random.randint(len(label))
                    relation_label[i, j] = label[rt]
                #开始设定没有关系的样本
                lower_bound = len(ins['na_triple'])
                # random.shuffle(ins['na_triple'])
                # lower_bound = max(20, len(train_tripe)*3)

                for j, (h_idx, t_idx) in enumerate(ins['na_triple'][:lower_bound], len(train_tripe)):
                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]

                    for h in hlist:
                        h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                    for t in tlist:
                        t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                    relation_multi_label[i, j, 0] = 1
                    relation_label[i, j] = 0
                    relation_mask[i, j] = 1
                    delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
                    if delta_dis < 0:
                        ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                    else:
                        ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)
            #这一个批次中输入样本的长度, tensor([340, 246, 225, 170, 149, 136, 132, 129])
            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            #最大长度340
            max_c_len = int(input_lengths.max())
            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'relation_label': relation_label[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'input_lengths': input_lengths,
                   'pos_idx': pos_idx[:cur_bsz, :max_c_len].contiguous(),
                   'relation_multi_label': relation_multi_label[:cur_bsz, :max_h_t_cnt],
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
                   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   }

    def get_test_batch(self):
        context_idxs = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_pos = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        h_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
        t_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
        context_ner = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_char_idxs = torch.LongTensor(self.test_batch_size, self.max_length, self.char_limit).cuda()
        relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).cuda()
        ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).cuda()

        for b in range(self.test_batches):
            start_id = b * self.test_batch_size
            cur_bsz = min(self.test_batch_size, self.test_len - start_id)
            cur_batch = list(self.test_order[start_id: start_id + cur_bsz])

            for mapping in [h_mapping, t_mapping, relation_mask]:
                mapping.zero_()

            ht_pair_pos.zero_()

            max_h_t_cnt = 1

            cur_batch.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)

            labels = []

            L_vertex = []
            titles = []
            indexes = []
            for i, index in enumerate(cur_batch):
                context_idxs[i].copy_(torch.from_numpy(self.data_test_word[index, :]))
                context_pos[i].copy_(torch.from_numpy(self.data_test_pos[index, :]))
                context_char_idxs[i].copy_(torch.from_numpy(self.data_test_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_test_ner[index, :]))

                idx2label = defaultdict(list)
                ins = self.test_file[index]

                for label in ins['labels']:
                    idx2label[(label['h'], label['t'])].append(label['r'])

                L = len(ins['vertexSet'])
                titles.append(ins['title'])

                j = 0
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            hlist = ins['vertexSet'][h_idx]
                            tlist = ins['vertexSet'][t_idx]

                            for h in hlist:
                                h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (
                                            h['pos'][1] - h['pos'][0])
                            for t in tlist:
                                t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (
                                            t['pos'][1] - t['pos'][0])

                            relation_mask[i, j] = 1

                            delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
                            if delta_dis < 0:
                                ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                            else:
                                ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
                            j += 1

                max_h_t_cnt = max(max_h_t_cnt, j)
                label_set = {}
                for label in ins['labels']:
                    label_set[(label['h'], label['t'], label['r'])] = label['in' + self.train_prefix]

                labels.append(label_set)

                L_vertex.append(L)
                indexes.append(index)

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'labels': labels,
                   'L_vertex': L_vertex,
                   'input_lengths': input_lengths,
                   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
                   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'titles': titles,
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'indexes': indexes
                   }

    def train(self, model_pattern, model_name):
        """
        训练模型
        :param model_pattern: 要初始化的模型, <class 'models.BiLSTM.BiLSTM'>
        :param model_name: 例如： 要保存的模型名称
        :return:
        """
        # 开始初始化模型
        print(f"初始化模型: {model_pattern.__name__}")
        ori_model = model_pattern(config=self)
        if self.pretrain_model != None:
            ori_model.load_state_dict(torch.load(self.pretrain_model))
        if self.use_gpu:
            # 如果使用gpu，那么放到gpu上
            ori_model.cuda()
        model = nn.DataParallel(ori_model)
        # 优化器参数
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        # nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
        BCE = nn.BCEWithLogitsLoss(reduction='none')

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        best_auc = 0.0
        best_f1 = 0.0
        best_epoch = 0

        model.train()

        global_step = 0
        total_loss = 0
        start_time = time.time()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.3, 1.0)
        plt.xlim(0.0, 0.4)
        plt.title('Precision-Recall')
        plt.grid(True)

        for epoch in trange(self.max_epoch, desc='Epoch: '):

            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            #获取一个batch的数据的特征
            for data in self.get_train_batch():

                context_idxs = data['context_idxs']
                context_pos = data['context_pos']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                relation_label = data['relation_label']
                input_lengths = data['input_lengths']
                relation_multi_label = data['relation_multi_label']
                relation_mask = data['relation_mask']
                context_ner = data['context_ner']
                context_char_idxs = data['context_char_idxs']
                ht_pair_pos = data['ht_pair_pos']
                # 尽量把它们的距离变成正数
                dis_h_2_t = ht_pair_pos + 10
                dis_t_2_h = -ht_pair_pos + 10
                #特征放入模型, predict_re [batch_size, max_entity_num, relations]
                predict_re = model(context_idxs, context_pos, context_ner, context_char_idxs, input_lengths, h_mapping,
                                   t_mapping, relation_mask, dis_h_2_t, dis_t_2_h)
                loss = torch.sum(BCE(predict_re, relation_multi_label) * relation_mask.unsqueeze(2)) / (
                            self.relation_num * torch.sum(relation_mask))

                output = torch.argmax(predict_re, dim=-1)
                output = output.data.cpu().numpy()

                optimizer.zero_grad()
                print(f"开始损失反向传播计算梯度")
                loss.backward()
                optimizer.step()

                relation_label = relation_label.data.cpu().numpy()

                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        label = relation_label[i][j]
                        if label < 0:
                            break

                        if label == 0:
                            self.acc_NA.add(output[i][j] == label)
                        else:
                            self.acc_not_NA.add(output[i][j] == label)

                        self.acc_total.add(output[i][j] == label)

                global_step += 1
                total_loss += loss.item()

                if global_step % self.period == 0:
                    cur_loss = total_loss / self.period
                    elapsed = time.time() - start_time
                    logging(
                        '| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:5.3f} | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(
                            epoch, global_step, elapsed * 1000 / self.period, cur_loss, self.acc_NA.get(),
                            self.acc_not_NA.get(), self.acc_total.get()))
                    total_loss = 0
                    start_time = time.time()

            if (epoch + 1) % self.test_epoch == 0:
                logging('-' * 89)
                eval_start_time = time.time()
                model.eval()
                f1, auc, pr_x, pr_y = self.test(model, model_name)
                model.train()
                logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
                logging('-' * 89)

                if f1 > best_f1:
                    best_f1 = f1
                    best_auc = auc
                    best_epoch = epoch
                    path = os.path.join(self.checkpoint_dir, model_name)
                    torch.save(ori_model.state_dict(), path)

                    plt.plot(pr_x, pr_y, lw=2, label=str(epoch))
                    plt.legend(loc="upper right")
                    plt.savefig(os.path.join("fig_result", model_name))

        print("Finish training")
        print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
        print("Storing best result...")
        print("Finish storing")

    def test(self, model, model_name, output=False, input_theta=-1, mode='test'):
        """

        :param model:
        :param model_name:
        :param output:
        :param input_theta:
        :param mode: 保存输出的命名使用
        :return:
        """
        data_idx = 0
        eval_start_time = time.time()
        # test_result_ignore = []
        total_recall_ignore = 0

        test_result = []
        total_recall = 0
        top1_acc = have_label = 0

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        for data in self.get_test_batch():
            with torch.no_grad():
                context_idxs = data['context_idxs']
                context_pos = data['context_pos']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                labels = data['labels']
                L_vertex = data['L_vertex']
                input_lengths = data['input_lengths']
                context_ner = data['context_ner']
                context_char_idxs = data['context_char_idxs']
                relation_mask = data['relation_mask']
                ht_pair_pos = data['ht_pair_pos']

                titles = data['titles']
                indexes = data['indexes']

                dis_h_2_t = ht_pair_pos + 10
                dis_t_2_h = -ht_pair_pos + 10

                predict_re = model(context_idxs, context_pos, context_ner, context_char_idxs, input_lengths,
                                   h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h)

                predict_re = torch.sigmoid(predict_re)

            predict_re = predict_re.data.cpu().numpy()

            for i in range(len(labels)):
                label = labels[i]
                index = indexes[i]

                total_recall += len(label)
                for l in label.values():
                    if not l:
                        total_recall_ignore += 1

                L = L_vertex[i]
                j = 0

                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            r = np.argmax(predict_re[i, j])
                            if (h_idx, t_idx, r) in label:
                                top1_acc += 1

                            flag = False

                            for r in range(1, self.relation_num):
                                intrain = False

                                if (h_idx, t_idx, r) in label:
                                    flag = True
                                    if label[(h_idx, t_idx, r)] == True:
                                        intrain = True

                                # if not intrain:
                                # 	test_result_ignore.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]),  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )

                                test_result.append(((h_idx, t_idx, r) in label, float(predict_re[i, j, r]), intrain,
                                                    titles[i], self.id2rel[r], index, h_idx, t_idx, r))

                            if flag:
                                have_label += 1

                            j += 1

            data_idx += 1

            if data_idx % self.period == 0:
                print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.period, (time.time() - eval_start_time)))
                eval_start_time = time.time()

        # test_result_ignore.sort(key=lambda x: x[1], reverse=True)
        test_result.sort(key=lambda x: x[1], reverse=True)

        print('total_recall', total_recall)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim(0.2, 1.0)
        # plt.xlim(0.0, 0.6)
        # plt.title('Precision-Recall')
        # plt.grid(True)

        pr_x = []
        pr_y = []
        correct = 0
        w = 0

        if total_recall == 0:
            total_recall = 1  # for test

        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        theta = test_result[f1_pos][1]

        if input_theta == -1:
            w = f1_pos
            input_theta = theta

        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        if not self.is_test:
            logging('ALL  : Theta {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, f1, auc))
        else:
            logging('ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta,
                                                                                                      f1_arr[w], auc))

        if output:
            # output = [x[-4:] for x in test_result[:w+1]]
            output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]} for x
                      in test_result[:w + 1]]
            json.dump(output, open(mode + "_index.json", "w"))

        # plt.plot(pr_x, pr_y, lw=2, label=model_name)
        # plt.legend(loc="upper right")
        if not os.path.exists(self.fig_result_dir):
            os.mkdir(self.fig_result_dir)
        # plt.savefig(os.path.join(self.fig_result_dir, model_name))

        pr_x = []
        pr_y = []
        correct = correct_in_train = 0
        w = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            if item[0] & item[2]:
                correct_in_train += 1
            if correct_in_train == correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            pr_y.append(p)
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()

        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)

        logging(
            'Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta,
                                                                                                     f1_arr[w], auc))

        return f1, auc, pr_x, pr_y

    def testall(self, model_pattern, model_name, input_theta):  # , ignore_input_theta):
        model = model_pattern(config=self)

        model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, model_name)))
        model.cuda()
        model.eval()
        f1, auc, pr_x, pr_y = self.test(model, model_name, True, input_theta)

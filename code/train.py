import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'BiLSTM', help = 'name of the model')
parser.add_argument('--save_name', type = str)

parser.add_argument('--do_train', action='store_true', help='训练模型')
parser.add_argument('--do_train_distant', action='store_true', help='默认训练标注的数据，如果训练远端生成的数据的模型，请加次参数')
parser.add_argument('--do_dev', action='store_true', help='评估模型')
parser.add_argument('--do_test', action='store_true', help='测试模型')
parser.add_argument('--epoch', type = int, default =200, help='训练多少个epoch')
parser.add_argument('--gpu', action='store_true', help='是否使用GPU，默认不使用')


args = parser.parse_args()
model = {
	'CNN3': models.CNN3,
	'LSTM': models.LSTM,
	'BiLSTM': models.BiLSTM,
	'ContextAware': models.ContextAware,
}

con = config.Config(args)

if args.do_train_distant:
	#远端监督的训练数据
	train_word_file = 'prepro_data/train_dis_word.npy'
	train_pos_file = 'prepro_data/train_dis_pos.npy'
	train_ner_file = 'prepro_data/train_dis_ner.npy'
	train_char_file = 'prepro_data/train_dis_char.npy'
	train_json_file = 'prepro_data/train_dis.json'
else:
	#人工标注的数据
	train_word_file = 'prepro_data/train_ano_word.npy'
	train_pos_file = 'prepro_data/train_ano_pos.npy'
	train_ner_file = 'prepro_data/train_ano_ner.npy'
	train_char_file = 'prepro_data/train_ano_char.npy'
	train_json_file = 'prepro_data/train_ano.json'

#评估数据
dev_word_file = 'prepro_data/dev_dev_word.npy'
dev_pos_file = 'prepro_data/dev_dev_word.npy'
dev_ner_file = 'prepro_data/dev_dev_ner.npy'
dev_char_file = 'prepro_data/dev_dev_char.npy'
dev_file = 'prepro_data/dev_dev.json'

#测试数据
test_word_file = 'prepro_data/dev_test_word.npy'
test_pos_file = 'prepro_data/dev_test_word.npy'
test_ner_file = 'prepro_data/dev_test_ner.npy'
test_char_file = 'prepro_data/dev_test_char.npy'
test_file = 'prepro_data/dev_test.json'

if args.do_train:
	con.load_train_data(train_word_file, train_pos_file, train_ner_file, train_char_file, train_json_file)
	con.train(model[args.model_name], args.save_name)
if args.do_dev:
	con.load_test_data(dev_word_file, dev_pos_file, dev_ner_file, dev_char_file, dev_file, mode='dev')
if args.do_test:
	con.load_test_data(test_word_file, test_pos_file, test_ner_file, test_char_file, test_file, mode='test')

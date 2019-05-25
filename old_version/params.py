# -*- coding: utf-8 -*-
class Params:
	'''
	Parameters of our model
	'''
	src_train = "../data/src-train.txt"
	tgt_train = "../data/tgt-train.txt"
	src_test = "../data/src-val.txt"
	tgt_test = "../data/tgt-val.txt"

	maxlen = 10
	batch_size = 32
	hidden_units = 512
	logdir = 'logdir'
	num_epochs = 250
	num_identical = 6
	num_heads = 8
	dropout = 0.1
	learning_rate = 0.0001
	word_limit_size = 20
	word_limit_lower = 3
	checkpoint = 'checkpoint'

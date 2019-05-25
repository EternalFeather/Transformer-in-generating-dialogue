# -*- coding: utf-8 -*-
class Params:
	"""
	Parameters of our model.
	"""
	project_name = 'demo'
	vocab_path = "dictionary/"
	src_train = "data/src-train.txt"
	tgt_train = "data/tgt-train.txt"
	src_test = "data/src-val.txt"
	tgt_test = "data/tgt-val.txt"

	train_record = "data/processed/{}/train.tfrecord".format(project_name)
	test_record = "data/processed/{}/val.tfrecord".format(project_name)
	logdir = 'logdir'
	ckpt_path = 'checkpoint/{}'.format(project_name)
	eval_log_path = 'result/{}'.format(project_name)

	rebuild_vocabulary = False

	maxlen = 12
	buffer_size = 10000
	batch_size = 128
	word_limit_size = 5

	d_model = 512
	dff = 2048
	num_epochs = 10
	num_block = 6
	num_heads = 8
	dropout_rate = 0.1
	smooth_epsilon = 0.1
	learning_rate = 1e-4
	learning_rate_warmup_steps = 4000
	beta_1 = 0.9
	beta_2 = 0.98
	epsilon = 1e-9
	batch_show_every = 500
	epoch_show_every = 1
	plot_pos_embedding = False

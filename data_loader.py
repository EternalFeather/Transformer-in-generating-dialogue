# -*- coding: utf-8 -*-
from __future__ import print_function
from params import Params as pm
import codecs
import sys
import numpy as np
import tensorflow as tf

def load_vocab(vocab):
	'''
	Load word token from encoding dictionary

	Args:
		vocab: [String], vocabulary files
	''' 
	vocab = [line.split()[0] for line in codecs.open('dictionary/{}'.format(vocab), 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= pm.word_limit_size]
	word2idx_dic = {word: idx for idx, word in enumerate(vocab)}
	idx2word_dic = {idx: word for idx, word in enumerate(vocab)}
	return word2idx_dic, idx2word_dic

def generate_dataset(source_sents, target_sents):
	'''
	Parse source sentences and target sentences from corpus with some formats
	
	Parse word token of each sentences

	Args:
		source_sents: [List], encoding sentences from src-train file
		target_sents: [List], decoding sentences from tgt-train file

	Padding for word token sentence list
	'''
	en2idx, idx2en = load_vocab('en.vocab.tsv')
	de2idx, idx2de = load_vocab('de.vocab.tsv')

	in_list, out_list, Sources, Targets = [], [], [], []
	for source_sent, target_sent in zip(source_sents, target_sents):
		# 1 means <UNK>
		inpt = [en2idx.get(word, 1) for word in (source_sent + u" <EOS>").split()]
		outpt = [de2idx.get(word, 1) for word in (target_sent + u" <EOS>").split()]
		if max(len(inpt), len(outpt)) <= pm.maxlen:
			# sentence token list
			in_list.append(np.array(inpt))
			out_list.append(np.array(outpt))
			# sentence list
			Sources.append(source_sent)
			Targets.append(target_sent)

	X = np.zeros([len(in_list), pm.maxlen], np.int32)
	Y = np.zeros([len(out_list), pm.maxlen], np.int32)
	for i, (x, y) in enumerate(zip(in_list, out_list)):
		X[i] = np.lib.pad(x, (0, pm.maxlen - len(x)), 'constant', constant_values = (0, 0))
		Y[i] = np.lib.pad(y, (0, pm.maxlen - len(y)), 'constant', constant_values = (0, 0))

	return X, Y, Sources, Targets

def load_data(l_data):
	'''
	Read train-data from input datasets

	Args:
		l_data: [String], the file name of datasets which used to generate tokens
	'''
	if l_data == 'train':
		en_sents = [line for line in codecs.open(pm.src_train, 'r', 'utf-8').read().split('\n') if line]
		de_sents = [line for line in codecs.open(pm.tgt_train, 'r', 'utf-8').read().split('\n') if line]
		if len(en_sents) == len(de_sents):
			inpt, outpt, Sources, Targets = generate_dataset(en_sents, de_sents)
		else:
			print("MSG : Source length is different from Target length.")
			sys.exit(0)
		return inpt, outpt
	elif l_data == 'test':
		en_sents = [line for line in codecs.open(pm.src_test, 'r', 'utf-8').read().split('\n') if line]
		de_sents = [line for line in codecs.open(pm.tgt_test, 'r', 'utf-8').read().split('\n') if line]
		if len(en_sents) == len(de_sents):
			inpt, outpt, Sources, Targets = generate_dataset(en_sents, de_sents)
		else:
			print("MSG : Source length is different from Target length.")
			sys.exit(0)
		return inpt, Sources, Targets
	else:
		print("MSG : Error when load data.")
		sys.exit(0)

def get_batch_data():
	'''
	A batch dataset generator
	'''
	inpt, outpt = load_data("train")

	batch_num = len(inpt) // pm.batch_size

	inpt = tf.convert_to_tensor(inpt, tf.int32)
	outpt = tf.convert_to_tensor(outpt, tf.int32)

	# parsing data into queue used for pipeline operations as a generator. 
	input_queues = tf.train.slice_input_producer([inpt, outpt])

	# multi-thread processing using batch
	x, y = tf.train.shuffle_batch(input_queues,
								num_threads = 8,
								batch_size = pm.batch_size,
								capacity = pm.batch_size * 64,
								min_after_dequeue = pm.batch_size * 32,
								allow_smaller_final_batch = False)

	return x, y, batch_num




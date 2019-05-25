# -*- coding: utf-8 -*-
from __future__ import print_function
from params import Params as pm
import os
from collections import Counter
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt


def build_vocab(path, fname):
	"""
	Constructs vocabulary as a dictionary.

	Args:
		:param path: [String], Input file path
		:param fname: [String], Output file name
	"""
	words = open(path, 'r', encoding='utf-8').read().split()
	wordCount = Counter(words)
	if not os.path.exists(pm.vocab_path):
		os.makedirs(pm.vocab_path)
	with open(pm.vocab_path + fname, 'w', encoding='utf-8') as f:
		f.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<SOS>", "<EOS>"))
		for word, count in wordCount.most_common(len(wordCount)):
			f.write(u"{}\t{}\n".format(word, count))


def load_vocab(vocab):
	"""
	Load word token from encoding dictionary.

	Args:
		:param vocab: [String], vocabulary files
		:return: tokenizer
	"""
	vocab = [line.split()[0] for line in open(
		'{}{}'.format(pm.vocab_path, vocab), 'r', encoding='utf-8').read().splitlines()
		if int(line.split()[1]) >= pm.word_limit_size]
	word2idx_dic = {word: idx for idx, word in enumerate(vocab)}
	idx2word_dic = {idx: word for idx, word in enumerate(vocab)}
	return word2idx_dic, idx2word_dic


if not os.path.exists(pm.vocab_path) or pm.rebuild_vocabulary:
	build_vocab(pm.src_train, "en.vocab.tsv")
	build_vocab(pm.tgt_train, "de.vocab.tsv")
en2idx, idx2en = load_vocab("en.vocab.tsv")
de2idx, idx2de = load_vocab("de.vocab.tsv")


def tokenize_sequences(source_sent, target_sent):
	"""
	Parse source sentences and target sentences from corpus with some formats.
	Parse word token from each sentences.
	Padding for word token sentence list.

	Args:
		:param source_sent: [List], encoding sentences from src-train file
		:param target_sent: [List], decoding sentences from tgt-train file
		:return: token sequences & source sentences
	"""
	source_sent = source_sent.numpy().decode('utf-8')
	target_sent = target_sent.numpy().decode('utf-8')

	if len(source_sent.split()) > pm.maxlen - 2:
		source_sent = source_sent[: pm.maxlen - 2]
	if len(target_sent.split()) > pm.maxlen - 2:
		target_sent = target_sent[: pm.maxlen - 2]

	inpt = [en2idx.get(word, 1) for word in (u"<SOS> " + source_sent + u" <EOS>").split()]
	outpt = [de2idx.get(word, 1) for word in (u"<SOS> " + target_sent + u" <EOS>").split()]

	return inpt, outpt


def jit_tokenize_sequences(source_sent, target_sent):
	return tf.py_function(tokenize_sequences, [source_sent, target_sent], [tf.int64, tf.int64])


def filter_single_word(source_sent, target_sent):
	return tf.logical_and(tf.size(source_sent) > 1, tf.size(target_sent) > 1)


def _byte_features(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def dump2record(filename, corpus1, corpus2):
	"""
	Writedown the data into tfrecord format.

	Args:
		:param filename:
		:param corpus1:
		:param corpus2:
	"""
	assert len(corpus1) == len(corpus2)
	writer = tf.io.TFRecordWriter(filename)

	for sent1, sent2 in tqdm(zip(corpus1, corpus2)):
		features = {}
		features['src_sent'] = _byte_features(sent1.encode('utf-8'))
		features['tgt_sent'] = _byte_features(sent2.encode('utf-8'))

		tf_features = tf.train.Features(feature=features)
		tf_examples = tf.train.Example(features=tf_features)
		tf_serialized = tf_examples.SerializeToString()

		writer.write(tf_serialized)

	writer.close()


def build_dataset(mode, filename=None, corpus=None, is_training=True):
	"""
	Read train-data from input datasets.

	Args:
		:param mode: [String], the tfrecord load mode, including 'array'(load from array) or 'file'(load from file)
		:param filename: [String], if mode == 'file' then input the path of tfrecord
		:param corpus: [String], if mode == 'array' then input the corpus with array type
		:return: datasets
	"""
	if mode == 'array':
		assert corpus is not None
		dataset = tf.data.Dataset.from_tensor_slices([sent.encode('utf-8') for sent in corpus])
		dataset = dataset.map(jit_tokenize_sequences)
		dataset = dataset.filter(filter_single_word).cache().shuffle(pm.buffer_size) if is_training else dataset
		dataset = dataset.padded_batch(pm.batch_size, padded_shapes=([-1], [-1])) if is_training else \
			dataset.padded_batch(1, padded_shapes=([-1], [-1]))
		dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) if is_training else dataset
		return dataset
	elif mode == 'file':
		def _parse(example):
			dics = {
				'src_sent': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),
				'tgt_sent': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None)
			}

			parsed_data = tf.io.parse_single_example(example, dics)
			src_sent = parsed_data['src_sent']
			tgt_sent = parsed_data['tgt_sent']
			return src_sent, tgt_sent

		assert filename is not None
		dataset = tf.data.TFRecordDataset(filename)
		dataset = dataset.map(_parse)
		dataset = dataset.map(jit_tokenize_sequences)
		dataset = dataset.filter(filter_single_word).cache().shuffle(pm.buffer_size) if is_training else dataset
		dataset = dataset.padded_batch(pm.batch_size, padded_shapes=([-1], [-1])) if is_training else \
			dataset.padded_batch(1, padded_shapes=([-1], [-1]))
		dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) if is_training else dataset
		return dataset
	else:
		raise ValueError('Something wrong about the mode when loading dataset ...')


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=4000):
		super(LRSchedule, self).__init__()

		# It must be tensor else raise "Could not find valid device for node." error.
		self.d_model = tf.cast(d_model, tf.float32)
		self.warmup_steps = warmup_steps

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)

		return tf.math.rsqrt(self.d_model) * tf.minimum(arg1, arg2)


def masking(sequence, task='padding'):
	"""
	Masking operation.

	Args:
		:param sequence: [Tensor], A tensor contains the ids to be search from the lookup table, shape = [batch_size, seq_len]
		:param task: [String], 'padding' or 'look_ahead' tasks, set 'padding' default
		:return: [Tensor], Masked matrix
	"""
	if task == 'padding':
		return tf.cast(tf.math.equal(sequence, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

	elif task == 'look_ahead':
		size = tf.shape(sequence)[1]
		return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

	else:
		raise ValueError('Please check the tasks that masking operation dealing with ("padding" or "look_ahead")...')


def create_masks(inp, tar):
	enc_padding_mask = masking(inp, task='padding')
	dec_padding_mask = masking(inp, task='padding')

	look_ahead_mask = masking(tar, task='look_ahead')
	dec_tar_padding_mask = masking(tar, task='padding')
	combined_mask = tf.maximum(dec_tar_padding_mask, look_ahead_mask)

	return enc_padding_mask, combined_mask, dec_padding_mask


def plot_attention_weights(attention, sentence, result, block):
	fig = plt.figure(figsize=(16, 8))

	sentence = [en2idx.get(word, 1) for word in sentence]
	attention = tf.squeeze(attention[block], axis=0)

	for head in range(attention.shape[0]):
		ax = fig.add_subplot(2, 4, head + 1)

		ax.matshow(attention[head][:-1, :], cmap='viridis')
		fontdict = {'fontsize': 10}

		ax.set_xtricks(range(len(sentence) + 2))
		ax.set_yticks(range(len(result)))

		ax.set_ylim(len(result)-1.5, -0.5)

		ax.set_xticklabels(['<SOS>'] + [list(idx2en.get(i, 1)) for i in sentence] + ['<EOS>'],
						   fontdict=fontdict, rotation=90)

		ax.set_yticklabels([list(idx2en.get(i, 1)) for i in result if i < len(idx2en)],
						   fontdict=fontdict)

		ax.set_xlabel('Head {}'.format(head + 1))

	plt.tight_layout()
	plt.show()



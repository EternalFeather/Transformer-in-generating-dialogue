# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
from params import Params as pm
from data_loader import get_batch_data, load_vocab
from modules import *
from tqdm import tqdm
import os

class Graph():
	def __init__(self, is_training = True):
		self.graph = tf.Graph()
		with self.graph.as_default():
			if is_training:
				self.inpt, self.outpt, self.batch_num = get_batch_data()
			else:
				self.inpt = tf.placeholder(tf.int32, shape = (None, pm.maxlen))
				self.outpt = tf.placeholder(tf.int32, shape = (None, pm.maxlen))

			# start with 2(<STR>) and without 3(<EOS>)
			self.decoder_input = tf.concat((tf.ones_like(self.outpt[:, :1])*2, self.outpt[:, :-1]), -1)

			en2idx, idx2en = load_vocab('en.vocab.tsv')
			de2idx, idx2de = load_vocab('de.vocab.tsv')

			# Encoder
			with tf.variable_scope("encoder"):
				self.enc = embedding(self.inpt,
									vocab_size = len(en2idx),
									num_units = pm.hidden_units,
									scale = True,
									scope = "enc_embed")

				# Position Encoding(use range from 0 to len(inpt) to represent position dim)
				self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.inpt)[1]), 0), [tf.shape(self.inpt)[0], 1]),
									vocab_size = pm.maxlen,
									num_units = pm.hidden_units,
									zero_pad = False,
									scale = False,
									scope = "enc_pe")

				# Dropout
				self.enc = tf.layers.dropout(self.enc,
											rate = pm.dropout,
											training = tf.convert_to_tensor(is_training))

				# Identical
				for i in range(pm.num_identical):
					with tf.variable_scope("num_identical_{}".format(i)):
						# Multi-head Attention
						self.enc = multihead_attention(queries = self.enc,
														keys = self.enc,
														num_units = pm.hidden_units,
														num_heads = pm.num_heads,
														dropout_rate = pm.dropout,
														is_training = is_training,
														causality = False)

						self.enc = feedforward(self.enc, num_units = [4 * pm.hidden_units, pm.hidden_units])

			# Decoder
			with tf.variable_scope("decoder"):
				self.dec = embedding(self.decoder_input,
								vocab_size = len(de2idx),
								num_units = pm.hidden_units,
								scale = True,
								scope = "dec_embed")

				# Position Encoding(use range from 0 to len(inpt) to represent position dim)
				self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_input)[1]), 0), [tf.shape(self.decoder_input)[0], 1]),
									vocab_size = pm.maxlen,
									num_units = pm.hidden_units,
									zero_pad = False,
									scale = False,
									scope = "dec_pe")

				# Dropout
				self.dec = tf.layers.dropout(self.dec,
											rate = pm.dropout,
											training = tf.convert_to_tensor(is_training))

				# Identical
				for i in range(pm.num_identical):
					with tf.variable_scope("num_identical_{}".format(i)):
						# Multi-head Attention(self-attention)
						self.dec = multihead_attention(queries = self.dec,
														keys = self.dec,
														num_units = pm.hidden_units,
														num_heads = pm.num_heads,
														dropout_rate = pm.dropout,
														is_training = is_training,
														causality = True,
														scope = "self_attention")

						# Multi-head Attention(vanilla-attention)
						self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.enc, 
                                                        num_units=pm.hidden_units, 
                                                        num_heads=pm.num_heads,
                                                        dropout_rate=pm.dropout,
                                                        is_training=is_training, 
                                                        causality=False,
                                                        scope="vanilla_attention")

						self.dec = feedforward(self.dec, num_units = [4 * pm.hidden_units, pm.hidden_units])

			# Linear
			self.logits = tf.layers.dense(self.dec, len(de2idx))
			self.preds = tf.to_int32(tf.arg_max(self.logits, dimension = -1))
			self.istarget = tf.to_float(tf.not_equal(self.outpt, 0))
			self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.outpt)) * self.istarget) / (tf.reduce_sum(self.istarget))
			tf.summary.scalar('acc', self.acc)

			if is_training:
				# smooth inputs
				self.y_smoothed = label_smoothing(tf.one_hot(self.outpt, depth = len(de2idx)))
				# loss function
				self.loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.y_smoothed)
				self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

				self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
				# optimizer
				self.optimizer = tf.train.AdamOptimizer(learning_rate = pm.learning_rate, beta1 = 0.9, beta2 = 0.98, epsilon = 1e-8)
				self.train_op = self.optimizer.minimize(self.mean_loss, global_step = self.global_step)

				tf.summary.scalar('mean_loss', self.mean_loss)
				self.merged = tf.summary.merge_all()

if __name__ == '__main__':
	en2idx, idx2en = load_vocab('en.vocab.tsv')
	de2idx, idx2de = load_vocab('de.vocab.tsv')

	g = Graph("train")
	print("MSG : Graph loaded!")

	# save model and use this model to training
	supvisor = tf.train.Supervisor(graph = g.graph,
									logdir = pm.logdir,
									save_model_secs = 0)

	with supvisor.managed_session() as sess:
		for epoch in range(1, pm.num_epochs + 1):
			if supvisor.should_stop():
				break
			# process bar
			for step in tqdm(range(g.batch_num), total = g.batch_num, ncols = 70, leave = False, unit = 'b'):
				sess.run(g.train_op)

			if not os.path.exists(pm.checkpoint):
				os.mkdir(pm.checkpoint)
			g_step = sess.run(g.global_step)
			supvisor.saver.save(sess, pm.checkpoint + '/model_epoch_%02d_gs_%d' % (epoch, g_step))

	print("MSG : Done!")


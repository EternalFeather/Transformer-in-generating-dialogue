# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def positional_encoding(seq_len, num_units, visualization=False):
	"""
	Positional_Encoding for a given tensor.

	Args:
		:param inputs: [Tensor], A tensor contains the ids to be search from the lookup table, shape = [batch_size, seq_len]
		:param num_units: [Int], Hidden size of embedding
		:param visualization: [Boolean], If True, it will plot the graph of position encoding
		:return: [Tensor] A tensor with shape [1, seq_len, num_units]
	"""
	def __get_angles(pos, i, d_model):
		angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
		return pos * angle_rates

	angle_rads = __get_angles(np.arange(seq_len)[:, np.newaxis],
							  np.arange(num_units)[np.newaxis, :],
							  num_units)

	sine = np.sin(angle_rads[:, 0::2])
	cosine = np.cos(angle_rads[:, 1::2])

	pos_encoding = np.concatenate([sine, cosine], axis=-1)
	pos_encoding = pos_encoding[np.newaxis, ...]

	if visualization:
		plt.figure(figsize=(12, 8))
		plt.pcolormesh(pos_encoding[0], cmap='RdBu')
		plt.xlabel('Depth')
		plt.xlim((0, num_units))
		plt.ylabel('Position')
		plt.colorbar()
		plt.show()

	return tf.cast(pos_encoding, tf.float32)


def scaled_dot_product_attention(q, k, v, mask=None):
	"""
	Calculate the attention weights.

	Args:
		:param q: [Tensor], query with shape [..., seq_len_q, d_model]
		:param k: [Tensor], key with shape [..., seq_len_k, d_model]
		:param v: [Tensor], value with shape [..., seq_len_v, d_model]
		:param mask: [Tensor], Float tensor with shape [..., seq_len_q, seq_len_k], default to None
		:return: [Tensor], output, attention_weights
	"""
	matmul_qk = tf.matmul(q, k, transpose_b=True)

	dk = tf.cast(tf.shape(k)[-1], tf.float32)
	scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

	# Heuristic mask implementation that add an infinitesimal number so that its effect can be ignored
	if mask is not None:
		scaled_attention_logits += (mask * -1e9)

	attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
	output = tf.matmul(attention_weights, v)

	return output, attention_weights


class multihead_attention(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads):
		super(multihead_attention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_moccccdel

		assert d_model % self.num_heads == 0
		self.depth = d_model // num_heads

		self.wq = tf.keras.layers.Dense(d_model)
		self.wk = tf.keras.layers.Dense(d_model)
		self.wv = tf.keras.layers.Dense(d_model)

		self.dense = tf.keras.layers.Dense(d_model)

	def split_heads(self, x, batch_size):
		"""
		Split the last dimension into (num_heads, depth).
		Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth).
        """

		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3])

	def call(self, v, k, q, mask):
		batch_size = tf.shape(q)[0]

		q = self.wq(q)
		k = self.wk(k)
		v = self.wv(v)

		q = self.split_heads(q, batch_size)
		k = self.split_heads(k, batch_size)
		v = self.split_heads(v, batch_size)

		scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
		concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
		output = self.dense(concat_attention)

		return output, attention_weights


class pointwise_feedforward(tf.keras.layers.Layer):
	def __init__(self, d_model, dff):
		super(pointwise_feedforward, self).__init__()
		self.d_model = d_model
		self.dff = dff

		self.dense_layer_1 = tf.keras.layers.Dense(dff, activation='relu')
		self.dense_layer_2 = tf.keras.layers.Dense(d_model)

	def call(self, x):
		output = self.dense_layer_1(x)
		output = self.dense_layer_2(output)

		return output


class EncoderBlock(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(EncoderBlock, self).__init__()
		self.multi_attn = multihead_attention(d_model, num_heads)
		self.ffn = pointwise_feedforward(d_model, dff)

		self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		self.dropout_1 = tf.keras.layers.Dropout(rate)
		self.dropout_2 = tf.keras.layers.Dropout(rate)

	def call(self, x, training, padding_mask):
		attn_output, attn_weight = self.multi_attn(x, x, x, padding_mask)
		attn_output = self.dropout_1(attn_output, training=training)
		output_1 = self.layer_norm_1(x + attn_output)

		ffn_output = self.ffn(output_1)
		ffn_output = self.dropout_2(ffn_output, training=training)
		output_2 = self.layer_norm_2(output_1 + ffn_output)

		return output_2, attn_weight


class DecoderBlock(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(DecoderBlock, self).__init__()
		self.multi_attn_1 = multihead_attention(d_model, num_heads)
		self.multi_attn_2 = multihead_attention(d_model, num_heads)

		self.ffn = pointwise_feedforward(d_model, dff)

		self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		self.dropout_1 = tf.keras.layers.Dropout(rate)
		self.dropout_2 = tf.keras.layers.Dropout(rate)
		self.dropout_3 = tf.keras.layers.Dropout(rate)

	def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
		attn_output_1, attn_weight_1 = self.multi_attn_1(x, x, x, look_ahead_mask)
		attn_output_1 = self.dropout_1(attn_output_1, training=training)
		output_1 = self.layer_norm_1(x + attn_output_1)

		attn_output_2, attn_weight_2 = self.multi_attn_2(enc_output, enc_output, output_1, padding_mask)
		attn_output_2 = self.dropout_2(attn_output_2, training=training)
		output_2 = self.layer_norm_2(output_1 + attn_output_2)

		ffn_output = self.ffn(output_2)
		ffn_output = self.dropout_3(ffn_output, training=training)
		output_3 = self.layer_norm_3(output_2 + ffn_output)

		return output_3, attn_weight_1, attn_weight_2


class Encoder(tf.keras.layers.Layer):
	def __init__(self, num_blocks, d_model, num_heads, dff, input_vocab_size, plot_pos_embedding, rate=0.1):
		super(Encoder, self).__init__()
		self.d_model = d_model
		self.num_blocks = num_blocks
		self.plot_pos_embedding = plot_pos_embedding

		self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
		self.pos_embedding = positional_encoding(input_vocab_size, d_model, plot_pos_embedding)

		self.enc_blocks = [EncoderBlock(d_model, num_heads, dff, rate) for _ in range(num_blocks)]
		self.dropout = tf.keras.layers.Dropout(rate)

	def call(self, x, training, padding_mask, attn_dict):
		seq_len = tf.shape(x)[1]

		x = self.embedding(x)
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
		x += self.pos_embedding[:, :seq_len, :]

		x = self.dropout(x, training=training)

		for i in range(self.num_blocks):
			x, attn_weight = self.enc_blocks[i](x, training, padding_mask)
			attn_dict['encoder_layer{}_block'.format(i + 1)] = attn_weight

		return x, attn_dict


class Decoder(tf.keras.layers.Layer):
	def __init__(self, num_blocks, d_model, num_heads, dff, target_vocab_size, plot_pos_embedding, rate=0.1):
		super(Decoder, self).__init__()
		self.d_model = d_model
		self.num_blocks = num_blocks
		self.plot_pos_embedding = plot_pos_embedding

		self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
		self.pos_embedding = positional_encoding(target_vocab_size, d_model, plot_pos_embedding)

		self.dec_blocks = [DecoderBlock(d_model, num_heads, dff, rate) for _ in range(num_blocks)]
		self.dropout = tf.keras.layers.Dropout(rate)

	def call(self, x, enc_output, training, look_ahead_mask, padding_mask, attn_dict):
		seq_len = tf.shape(x)[1]

		x = self.embedding(x)
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
		x += self.pos_embedding[:, :seq_len, :]

		x = self.dropout(x, training=training)

		for i in range(self.num_blocks):
			x, attn_weight_1, attn_weight_2 = self.dec_blocks[i](x, enc_output, training, look_ahead_mask, padding_mask)
			attn_dict['decoder_layer{}_block'.format(i + 1)] = attn_weight_1
			attn_dict['decoder_layer{}_cross'.format(i + 1)] = attn_weight_2

		return x, attn_dict


class Transformer(tf.keras.Model):
	def __init__(self, num_blocks, d_model, num_heads, dff, input_vocab_size, target_vocab_size, plot_pos_embedding, rate=0.1):
		super(Transformer, self).__init__()

		self.encoder = Encoder(num_blocks, d_model, num_heads, dff, input_vocab_size, plot_pos_embedding, rate)
		self.decoder = Decoder(num_blocks, d_model, num_heads, dff, target_vocab_size, plot_pos_embedding, rate)
		self.final_layer = tf.keras.layers.Dense(target_vocab_size)

	def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
		attn_dict = {}

		enc_output, attn_dict = self.encoder(inp, training, enc_padding_mask, attn_dict)
		dec_output, attn_dict = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask, attn_dict)
		final_output = self.final_layer(dec_output)

		return final_output, attn_dict

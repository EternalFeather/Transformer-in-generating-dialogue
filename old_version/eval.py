# -*- coding: utf-8 -*-
from __future__ import print_function
import codecs
import os
from train import Graph
from params import Params as pm
from data_loader import load_data, load_vocab
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import corpus_bleu


def eval():
	g = Graph(is_training = False)
	print("MSG : Graph loaded!")

	X, Sources, Targets = load_data('test')
	en2idx, idx2en = load_vocab('en.vocab.tsv')
	de2idx, idx2de = load_vocab('de.vocab.tsv')

	with g.graph.as_default():
		sv = tf.train.Supervisor()
		with sv.managed_session(config = tf.ConfigProto(allow_soft_placement = True)) as sess:
			# load pre-train model
			sv.saver.restore(sess, tf.train.latest_checkpoint(pm.checkpoint))
			print("MSG : Restore Model!")

			mname = open(pm.checkpoint + '/checkpoint', 'r').read().split('"')[1]

			if not os.path.exists('Results'):
				os.mkdir('Results')
			with codecs.open("Results/" + mname, 'w', 'utf-8') as f:
				list_of_refs, predict = [], []
				# Get a batch
				for i in range(len(X) // pm.batch_size):
					x = X[i * pm.batch_size: (i + 1) * pm.batch_size]
					sources = Sources[i * pm.batch_size: (i + 1) * pm.batch_size]
					targets = Targets[i * pm.batch_size: (i + 1) * pm.batch_size]

					# Autoregressive inference
					preds = np.zeros((pm.batch_size, pm.maxlen), dtype = np.int32)
					for j in range(pm.maxlen):
						_preds = sess.run(g.preds, feed_dict = {g.inpt: x, g.outpt: preds})
						preds[:, j] = _preds[:, j]

					for source, target, pred in zip(sources, targets, preds):
						got = " ".join(idx2de[idx] for idx in pred).split("<EOS>")[0].strip()
						f.write("- Source: {}\n".format(source))
						f.write("- Ground Truth: {}\n".format(target))
						f.write("- Predict: {}\n\n".format(got))
						f.flush()

						# Bleu Score
						ref = target.split()
						prediction = got.split()
						if len(ref) > pm.word_limit_lower and len(prediction) > pm.word_limit_lower:
							list_of_refs.append([ref])
							predict.append(prediction)

				score = corpus_bleu(list_of_refs, predict)
				f.write("Bleu Score = " + str(100 * score))


if __name__ == '__main__':
	eval()
	print("MSG : Done!")

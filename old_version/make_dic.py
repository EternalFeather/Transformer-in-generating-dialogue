# -*- coding: utf-8 -*-
from __future__ import print_function
from params import Params as pm
import codecs
import os
from collections import Counter


def make_dic(path, fname):
	'''
	Constructs vocabulary as a dictionary

	Args:
		path: [String], Input file path
		fname: [String], Output file name

	Build vocabulary line by line to dictionary/ path
	'''
	text = codecs.open(path, 'r', 'utf-8').read()
	words = text.split()
	wordCount = Counter(words)
	if not os.path.exists('dictionary'):
		os.mkdir('dictionary')
	with codecs.open('dictionary/{}'.format(fname), 'w', 'utf-8') as f:
		f.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>","<UNK>","<STR>","<EOS>"))
		for word, count in wordCount.most_common(len(wordCount)):
			f.write(u"{}\t{}\n".format(word, count))


if __name__ == '__main__':
	make_dic(pm.src_train, "en.vocab.tsv")
	make_dic(pm.tgt_train, "de.vocab.tsv")
	print("MSG : Constructing Dictionary Finished!")


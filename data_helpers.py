import os
import re
import sys
import json
import pickle
import itertools
import numpy as np
import pandas as pd
import gensim as gs
from pprint import pprint
from collections import Counter
from tensorflow.contrib import learn

def clean_str(s):
	s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
	s = re.sub(r" : ", ":", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	return s.strip().lower()

def load_trained_vecs(vocabulary):
	model = gs.models.Word2Vec.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
	trained_vecs = {}
	for word in vocabulary:
		if word in model:
			trained_vecs[word] = model[word]
	return trained_vecs

def add_unknown_words(word_vecs, vocab, min_df=0, k=300):
	for word in vocab:
		if word not in word_vecs and vocab[word] >= min_df:
			word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

def pad_sentences(sentences, padding_word="<PAD/>", params=None):
	"""Padding setences during training or prediction"""
	if params is None: # Train
		sequence_length = max(len(x) for x in sentences)
	else: # Prediction
		print('This is prediction, reading the trained sequence length')
		sequence_length = params['sequence_length']

	print('The maximum length is {}'.format(sequence_length))
	padded_sentences = []
	for i in range(len(sentences)):
		sentence = sentences[i]
		num_padding = sequence_length - len(sentence)

		if num_padding < 0: # Prediction: cut off the sentence if it is longer than the sequence length
			print('This sentence has to be cut off because it is longer than trained sequence length')
			padded_sentence = sentence[0:sequence_length]
		else:
			padded_sentence = sentence + [padding_word] * num_padding

		padded_sentences.append(padded_sentence)
	return padded_sentences

def build_vocab(sentences):
	word_counts = Counter(itertools.chain(*sentences))
	vocabulary_inv = [x[0] for x in word_counts.most_common()]
	vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
	return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
	x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
	print("x are {}".format(x))
	y = np.array(labels)
	print("y are {}".format(y))
	return [x, y]

def batch_iter(data, batch_size, num_epochs, predict=False):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(len(data) / batch_size) + 1

	for epoch in range(num_epochs):
		if predict is False: # Shuffle the data during training and dev step
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else: # Do not shuffle the data during prediction step
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = (batch_num + 1) * batch_size
			if end_index > data_size:
				end_index = data_size
			yield shuffled_data[start_index:end_index]

def load_data_and_labels(filename):
	df = pd.read_csv(filename, sep='|')
	selected = ['PROPOSED_CATEGORY', 'DESCRIPTION_UNMASKED']
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1)
	df = df.dropna(axis=0, how='any', subset=selected)
	df = df.reindex(np.random.permutation(df.index))

	labels = sorted(list(set(df[selected[0]].tolist())))
	nb_labels = len(labels)
	one_hot = np.zeros((nb_labels, nb_labels), int)
	np.fill_diagonal(one_hot, 1)

	label_dict = dict(zip(labels, one_hot))
	for key in sorted(label_dict.keys()):
		print('{} ---> {}'.format(key, label_dict[key]))

	y = df[selected[0]].apply(lambda x: label_dict[x]).tolist()
	examples = df[selected[1]].apply(lambda x: clean_str(x).split(' ')).tolist()
	print(len(y))
	print(len(examples))
	for i in range(10):
		print('{} ---> {}'.format(y[i], examples[i]))
	return [examples, y, df, labels]

def load_data(filename):
	sentences, labels, df, labels_list = load_data_and_labels(filename)
	sentences_padded = pad_sentences(sentences)
	vocabulary, vocabulary_inv = build_vocab(sentences_padded)
	x, y = build_input_data(sentences_padded, labels, vocabulary)
	return [x, y, vocabulary, vocabulary_inv, df, labels_list]

if __name__ == "__main__":
	train_file = './data/bank_debit/input_40000.csv'
	load_data(train_file)

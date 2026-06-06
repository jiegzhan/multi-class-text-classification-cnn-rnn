import logging
import os
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)

PAD_TOKEN = "<PAD/>"
UNK_TOKEN = "<UNK/>"
PAD_INDEX = 0
UNK_INDEX = 1


def set_seed(seed):
	"""Set random seeds for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)


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
	s = re.sub(r"\(", " ( ", s)
	s = re.sub(r"\)", " ) ", s)
	s = re.sub(r"\?", " ? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	return s.strip().lower()


def build_vocab(sentences):
	"""Build vocabulary with reserved PAD=0 and UNK=1 tokens."""
	word_counts = Counter(word for sentence in sentences for word in sentence)
	word_counts.pop(PAD_TOKEN, None)

	vocabulary_inv = [PAD_TOKEN, UNK_TOKEN]
	vocabulary_inv.extend(word for word, _ in word_counts.most_common())
	vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
	return vocabulary, vocabulary_inv


def map_word_to_index(examples, vocabulary):
	"""Map token strings to indices; unknown words map to UNK."""
	unk_index = vocabulary.get(UNK_TOKEN, UNK_INDEX)
	return [
		[vocabulary.get(word, unk_index) for word in example]
		for example in examples
	]


def pad_sentences(sentences, forced_sequence_length=None):
	"""Pad or truncate sentences to a fixed length."""
	if forced_sequence_length is None:
		sequence_length = max(len(sentence) for sentence in sentences)
	else:
		logging.info("Using trained sequence length: %s", forced_sequence_length)
		sequence_length = forced_sequence_length

	logging.info("Sequence length: %s", sequence_length)

	padded_sentences = []
	for sentence in sentences:
		if len(sentence) > sequence_length:
			logging.debug("Truncating sentence longer than sequence length")
			padded_sentences.append(sentence[:sequence_length])
		else:
			padded_sentences.append(sentence + [PAD_TOKEN] * (sequence_length - len(sentence)))
	return padded_sentences


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""Yield mini-batches without empty trailing batches."""
	data = np.array(data, dtype=object)
	data_size = len(data)
	if data_size == 0:
		return

	num_batches_per_epoch = (data_size + batch_size - 1) // batch_size

	for _ in range(num_epochs):
		if shuffle:
			indices = np.random.permutation(data_size)
			shuffled_data = data[indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			batch = shuffled_data[start_index:end_index]
			if len(batch) > 0:
				yield batch


def load_random_embeddings(vocabulary_inv, embedding_dim):
	"""Initialize embedding matrix with small random values."""
	embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocabulary_inv), embedding_dim)).astype(np.float32)
	# Keep PAD at index 0 with a zero vector for stable padding semantics.
	embedding_mat[PAD_INDEX] = 0.0
	return embedding_mat


def load_data(filename):
	"""Load training data from a zip-compressed CSV file."""
	df = pd.read_csv(filename, compression="zip")
	selected = ["Category", "Descript"]
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1)
	df = df.dropna(axis=0, how="any", subset=selected)
	df = df.reindex(np.random.permutation(df.index))

	labels = sorted(df[selected[0]].unique().tolist())
	num_labels = len(labels)
	one_hot = np.eye(num_labels, dtype=np.int32)
	label_dict = {label: one_hot[index] for index, label in enumerate(labels)}

	x_raw = df[selected[1]].apply(lambda text: clean_str(text).split(" ")).tolist()
	y_raw = df[selected[0]].apply(lambda label: label_dict[label]).tolist()

	x_raw = pad_sentences(x_raw)
	vocabulary, vocabulary_inv = build_vocab(x_raw)
	x = np.array(map_word_to_index(x_raw, vocabulary), dtype=np.int32)
	y = np.array(y_raw, dtype=np.float32)
	return x, y, vocabulary, vocabulary_inv, df, labels


def load_test_data(test_file, labels):
	"""Load pipe-delimited prediction data."""
	df = pd.read_csv(test_file, sep="|")
	select = ["Descript"]

	df = df.dropna(axis=0, how="any", subset=select)
	test_examples = df[select[0]].apply(lambda text: clean_str(text).split(" ")).tolist()

	label_dict = {label: index for index, label in enumerate(labels)}
	y_ = None
	if "Category" in df.columns:
		select.append("Category")
		y_ = df[select[1]].apply(lambda label: label_dict[label]).tolist()

	not_select = list(set(df.columns) - set(select))
	df = df.drop(not_select, axis=1)
	return test_examples, y_, df


def prepare_test_features(test_examples, vocabulary, sequence_length):
	"""Tokenize, pad, and index test examples."""
	padded = pad_sentences(test_examples, forced_sequence_length=sequence_length)
	indexed = map_word_to_index(padded, vocabulary)
	return np.asarray(indexed, dtype=np.int32)


def stratified_label_indices(y):
	"""Return class indices for stratified splitting."""
	return np.argmax(y, axis=1)

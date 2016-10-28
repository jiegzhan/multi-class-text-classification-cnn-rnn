import os
import sys
import json
import pickle
import shutil
import data_helpers
import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint
from cnnlstm import cnnlstm_class

def get_trained_params(trained_results_dir):
	params = json.loads(open(trained_results_dir + 'parameters.json').read())
	word_index = json.loads(open(trained_results_dir + 'word_index.json').read())
	labels = json.loads(open(trained_results_dir + 'labels.json').read())

	with open(trained_results_dir + 'embedding.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)

	return params, word_index, labels, embedding_mat

def get_test_data(test_file, labels):
	df = pd.read_csv(test_file, sep='|')
	selected = ['PROPOSED_CATEGORY', 'DESCRIPTION_UNMASKED']
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1)
	df = df.dropna(axis=0, how='any', subset=selected)

	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	y_ = df[selected[0]].apply(lambda x: label_dict[x]).tolist()
	test_examples = df[selected[1]].apply(lambda x: data_helpers.clean_str(x).split(' ')).tolist()

	return test_examples, y_, df

def convert_word_to_id(examples, word_index):
	x_ = []
	for example in examples:
		temp = []
		for word in example:
			if word in word_index:
				temp.append(word_index[word])
			else:
				temp.append(0)
		x_.append(temp)
	return x_

params, word_index, labels, embedding_mat = get_trained_params('./train_result/')
x_, y_, df = get_test_data('./train_result/df_test.csv', labels)
x_ = data_helpers.pad_sentences(x_, params=params)
x_ = convert_word_to_id(x_, word_index)

x_ = np.asarray(x_)
y_ = np.asarray(y_)

x_test = x_
y_test = y_

predicted_results_dir = './predict_result/'
if os.path.exists(predicted_results_dir):
	shutil.rmtree(predicted_results_dir)
	print('The old predict_result directory has been deleted')
os.makedirs(predicted_results_dir)

df.to_csv('./predict_result/df_test_2.csv', index=False, header=True)

with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement = params['allow_soft_placement'],
		log_device_placement = params['log_device_placement'])
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		lstm = cnnlstm_class(
			embedding_mat = embedding_mat,
			non_static = params['non_static'],
			lstm_type = params['lstm_type'],
			hidden_unit = params['hidden_unit'],
			sequence_length = len(x_test[0]),
			max_pool_size = params['max_pool_size'],
			filter_sizes = map(int, params['filter_sizes'].split(",")),
			num_filters = params['num_filters'],
			num_classes = len(labels),
			embedding_size = params['embedding_dim'],
			l2_reg_lambda = params['l2_reg_lambda'])

		def real_len(xb):
			return [np.ceil(np.argmin(i + [0])*1.0/params['max_pool_size']) for i in xb]

		def predict_step(x_batch, y_batch):
			feed_dict = {
				lstm.input_x: x_batch,
				lstm.input_y: y_batch,
				lstm.dropout_keep_prob: 1.0,
				lstm.batch_size: len(x_batch),
				lstm.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
				lstm.real_len: real_len(x_batch),
			}
			nb_correct, predictions = sess.run([lstm.nb_correct, lstm.predictions], feed_dict)
			return nb_correct, predictions

		checkpoint_file = params['checkpoint_path']
		saver = tf.train.Saver(tf.all_variables())
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)
		print('{} has been loaded'.format(checkpoint_file))

		batches = data_helpers.batch_iter_test(list(zip(x_test, y_test)), params['batch_size'], 1)

		total_correct, predict_labels = 0, []
		for batch in batches:
			x_batch, y_batch = zip(*batch)
			print('x_batch: {}, y_batch: {}'.format(len(x_batch), len(y_batch)))
			nb_correct, predictions = predict_step(x_batch, y_batch)
			total_correct += nb_correct
			print(predictions)
			print(nb_correct)
			for item in predictions:
				predict_labels.append(labels[item])

		df['Predicted'] = predict_labels
		cols = df.columns
		df.to_csv('./predict_result/final_2.csv', index=False, columns=sorted(cols, reverse=True))
		print('The number of total correct on test set is {}'.format(total_correct))

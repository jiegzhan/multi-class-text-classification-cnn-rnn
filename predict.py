import os
import sys
import json
import pickle
import shutil
import data_helpers
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn_lstm import TextCNNLSTM

def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	word_index = json.loads(open(trained_dir + 'words_index.json').read())
	labels = json.loads(open(trained_dir + 'labels.json').read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)

	return params, word_index, labels, embedding_mat

def load_test_data(test_file, labels):
	df = pd.read_csv(test_file, sep='|')
	select = ['DESCRIPTION_UNMASKED']

	df = df.dropna(axis=0, how='any', subset=select)
	test_examples = df[select[0]].apply(lambda x: data_helpers.clean_str(x).split(' ')).tolist()

	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	y_ = None
	if 'PROPOSED_CATEGORY' in df.columns:
		select.append('PROPOSED_CATEGORY')
		y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()

	not_select = list(set(df.columns) - set(select))
	df = df.drop(not_select, axis=1)

	return test_examples, y_, df

def map_word_to_index(examples, word_index):
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

def predict_unseen_data(test_file, trained_dir):
	params, word_index, labels, embedding_mat = load_trained_params(trained_dir)
	x_, y_, df = load_test_data(test_file, labels)
	x_ = data_helpers.pad_sentences(x_, params=params)
	x_ = map_word_to_index(x_, word_index)

	x_test, y_test = np.asarray(x_), None
	if y_ is not None:
		y_test = np.asarray(y_)

	predicted_dir = './predicted_results/'
	if os.path.exists(predicted_dir):
		shutil.rmtree(predicted_dir)
		print('The old predict_result directory {} has been deleted'.format(predicted_dir))
	os.makedirs(predicted_dir)

	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(
			allow_soft_placement = params['allow_soft_placement'],
			log_device_placement = params['log_device_placement'])
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn_lstm = TextCNNLSTM(
				embedding_mat = embedding_mat,
				non_static = params['non_static'],
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

			def predict_step(x_batch):
				feed_dict = {
					cnn_lstm.input_x: x_batch,
					cnn_lstm.dropout_keep_prob: 1.0,
					cnn_lstm.batch_size: len(x_batch),
					cnn_lstm.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_lstm.real_len: real_len(x_batch),
				}
				predictions = sess.run([cnn_lstm.predictions], feed_dict)
				return predictions

			checkpoint_file = trained_dir + 'best_model.ckpt'
			saver = tf.train.Saver(tf.all_variables())
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file[:-5]))
			saver.restore(sess, checkpoint_file)
			print('{} has been loaded'.format(checkpoint_file))

			batches = data_helpers.batch_iter(list(x_test), params['batch_size'], 1, predict=True)

			predictions, predict_labels = [], []
			for x_batch in batches:
				batch_predictions = predict_step(x_batch)[0]
				print(batch_predictions)
				for batch_prediction in batch_predictions:
					predictions.append(batch_prediction)
					predict_labels.append(labels[batch_prediction])

			df['PREDICTED'] = predict_labels
			columns = sorted(df.columns, reverse=True)
			df.to_csv(predicted_dir + 'predictions_all.csv', index=False, columns=columns, sep='|')

			if y_test is not None:
				y_test = np.array(np.argmax(y_test, axis=1))
				correct_predictions = float(sum(np.array(predictions) == y_test))
				print('The number of test examples is: {}'.format(len(y_test)))
				print('The number of correct predictions is: {}'.format(correct_predictions))
				print('{}% of the predictions are correct'.format(float(correct_predictions) * 100 / len(y_test)))

				df_correct = df[df['PREDICTED'] == df['PROPOSED_CATEGORY']]
				df_non_correct = df_non_correct = df[df['PREDICTED'] != df['PROPOSED_CATEGORY']]
				df_correct.to_csv(predicted_dir + 'predictions_correct.csv', index=False, columns=columns, sep='|')
				df_non_correct.to_csv(predicted_dir + 'predictions_non_correct.csv', index=False, columns=columns, sep='|')

if __name__ == '__main__':
	test_file = './data/bank_debit/3000.csv'
	test_file = './data/bank_debit/130000.csv'
	test_file = './trained_results/data_test.csv'
	trained_dir = './trained_results/'
	predict_unseen_data(test_file, trained_dir)

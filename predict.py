import os
import sys
import json
import pickle
import shutil
import data_helpers
import numpy as np
import pandas as pd
import tensorflow as tf
from cnnlstm import cnnlstm_class

def get_trained_params(trained_result_dir):
	params = json.loads(open(trained_result_dir + 'parameters.json').read())
	word_index = json.loads(open(trained_result_dir + 'word_index.json').read())
	labels = json.loads(open(trained_result_dir + 'labels.json').read())

	with open(trained_result_dir + 'embedding.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)

	return params, word_index, labels, embedding_mat

def get_test_data(test_file, labels):
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

def predict(test_file, trained_result_dir):
	params, word_index, labels, embedding_mat = get_trained_params(trained_result_dir)
	x_, y_, df = get_test_data(test_file, labels)
	x_ = data_helpers.pad_sentences(x_, params=params)
	x_ = convert_word_to_id(x_, word_index)

	x_test, y_test = np.asarray(x_), None
	if y_ is not None:
		y_test = np.asarray(y_)

	predict_result_dir = './predict_result/'
	if os.path.exists(predict_result_dir):
		shutil.rmtree(predict_result_dir)
		print('The old predict_result directory has been deleted')
	os.makedirs(predict_result_dir)

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

			def predict_step(x_batch):
				feed_dict = {
					lstm.input_x: x_batch,
					lstm.dropout_keep_prob: 1.0,
					lstm.batch_size: len(x_batch),
					lstm.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					lstm.real_len: real_len(x_batch),
				}
				predictions = sess.run([lstm.predictions], feed_dict)
				return predictions

			checkpoint_file = './train_result/best_model.ckpt'
			saver = tf.train.Saver(tf.all_variables())
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file[:-5]))
			saver.restore(sess, checkpoint_file)
			print('{} has been loaded'.format(checkpoint_file))

			batches = data_helpers.batch_iter_test(list(x_test), params['batch_size'], 1)

			predictions, predict_labels = [], []
			for x_batch in batches:
				batch_predictions = predict_step(x_batch)[0]
				print(batch_predictions)
				for batch_prediction in batch_predictions:
					predictions.append(batch_prediction)
					predict_labels.append(labels[batch_prediction])

			df['PREDICTED'] = predict_labels
			columns = sorted(df.columns, reverse=True)
			df.to_csv(predict_result_dir + 'prediction.csv', index=False, columns=columns, sep='|')

			if y_test is not None:
				y_test = np.array(np.argmax(y_test, axis=1))
				correct_predictions = float(sum(np.array(predictions) == y_test))
				print('The number of test examples is: {}'.format(len(y_test)))
				print('The number of correct predictions is: {}'.format(correct_predictions))
				print('{}% of the predictions are correct'.format(float(correct_predictions) * 100 / len(y_test)))

				df_correct = df[df['PREDICTED'] == df['PROPOSED_CATEGORY']]
				df_non_correct = df_non_correct = df[df['PREDICTED'] != df['PROPOSED_CATEGORY']]
				df_correct.to_csv(predict_result_dir + 'correct.csv', index=False, columns=columns, sep='|')
				df_non_correct.to_csv(predict_result_dir + 'non_correct.csv', index=False, columns=columns, sep='|')

if __name__ == '__main__':
	test_file = './3000.csv'
	# test_file = './130000.csv' # max_len is larger than trained sequence length
	# test_file = './train_result/df_test.csv'
	trained_result_dir = './train_result/'
	predict(test_file, trained_result_dir)

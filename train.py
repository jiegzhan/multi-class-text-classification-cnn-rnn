import os
import sys
import json
import time
import shutil
import pickle
import logging
import data_helpers
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn_lstm import TextCNNLSTM

logging.getLogger().setLevel(logging.INFO)

def train_cnn_lstm(input_file):
	params = json.loads(open('./training_config.json').read())

	x_, y_, vocabulary, vocabulary_inv, df, labels = data_helpers.load_data(input_file)
	trained_vecs = data_helpers.load_trained_vecs(vocabulary)
	data_helpers.add_unknown_words(trained_vecs, vocabulary)

	embedding_mat = [trained_vecs[p] for i, p in enumerate(vocabulary_inv)]
	embedding_mat = np.array(embedding_mat, dtype = np.float32)
	print(embedding_mat.shape)

	# Split the original dataset into train set and test set
	test_size = int(0.1 * len(x_))
	x, x_test = x_[:-test_size], x_[-test_size:]
	y, y_test = y_[:-test_size], y_[-test_size:]

	trained_dir = './trained_results/'
	df_train, df_test = df[:-test_size], df[-test_size:]
	if os.path.exists(trained_dir):
		shutil.rmtree(trained_dir)
		logging.critical('The old directory {} has been deleted'.format(trained_dir))
	os.makedirs(trained_dir)

	df_train.to_csv(trained_dir + 'data_train.csv', index=False, sep='|')
	df_test.to_csv(trained_dir + 'data_test.csv', index=False, sep='|')

	shuffle_indices = np.random.permutation(np.arange(len(y)))
	x_shuffled = x[shuffle_indices]
	y_shuffled = y[shuffle_indices]

	# Split the train set into train set and dev set
	dev_size = int(0.1 * len(x_shuffled))
	x_train, x_dev = x_shuffled[:-dev_size], x_shuffled[-dev_size:]
	y_train, y_dev = y_shuffled[:-dev_size], y_shuffled[-dev_size:]
	logging.critical('Train: {}, dev: {}, test: {}'.format(len(x_train), dev_size, test_size))

	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(
			allow_soft_placement=params['allow_soft_placement'],
			log_device_placement=params['log_device_placement'])
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn_lstm = TextCNNLSTM(
				embedding_mat = embedding_mat,
				non_static = params['non_static'],
				hidden_unit = params['hidden_unit'],
				sequence_length = x_.shape[1],
				max_pool_size = params['max_pool_size'],
				filter_sizes = map(int, params['filter_sizes'].split(",")),
				num_filters = params['num_filters'],
				num_classes = y_.shape[1],
				embedding_size = params['embedding_dim'],
				l2_reg_lambda = params['l2_reg_lambda'])

			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.RMSPropOptimizer(1e-3, decay = 0.9)
			grads_and_vars = optimizer.compute_gradients(cnn_lstm.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "train_checkpoints"))
			if os.path.exists(checkpoint_dir):
				shutil.rmtree(checkpoint_dir)
				logging.critical('The old checkpoint directory {} has beed deleted'.format(checkpoint_dir))
			os.makedirs(checkpoint_dir)
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")

			saver = tf.train.Saver(tf.all_variables())
			sess.run(tf.initialize_all_variables())

			def real_len(xb):
				return [np.ceil(np.argmin(i + [0])*1.0/params['max_pool_size']) for i in xb]

			def train_step(x_batch, y_batch):
				feed_dict = {
					cnn_lstm.input_x: x_batch,
					cnn_lstm.input_y: y_batch,
					cnn_lstm.dropout_keep_prob: params['dropout_keep_prob'],
					cnn_lstm.batch_size: len(x_batch),
					cnn_lstm.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_lstm.real_len: real_len(x_batch),
				}
				_, step, loss, accuracy = sess.run([train_op, global_step, cnn_lstm.loss, cnn_lstm.accuracy], feed_dict)

			def dev_step(x_batch, y_batch):
				feed_dict = {
					cnn_lstm.input_x: x_batch,
					cnn_lstm.input_y: y_batch,
					cnn_lstm.dropout_keep_prob: 1.0,
					cnn_lstm.batch_size: len(x_batch),
					cnn_lstm.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_lstm.real_len: real_len(x_batch),
				}
				step, loss, accuracy, num_correct, predictions = sess.run(
					[global_step, cnn_lstm.loss, cnn_lstm.accuracy, cnn_lstm.num_correct, cnn_lstm.predictions], feed_dict)
				return accuracy, loss, num_correct, predictions

			# Training starts
			train_batches = data_helpers.batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
			best_accuracy, best_at_stp = 0, 0

			# Train the model batch by batch
			for train_batch in train_batches:
				x_train_batch, y_train_batch = zip(*train_batch)
				train_step(x_train_batch, y_train_batch)
				current_step = tf.train.global_step(sess, global_step)

				# Evaluate on dev set (batch by batch) during training
				if current_step % params['evaluate_every'] == 0:
					dev_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)

					total_dev_correct = 0
					for dev_batch in dev_batches:
						x_dev_batch, y_dev_batch = zip(*dev_batch)
						acc, loss, num_dev_correct, predictions = dev_step(x_dev_batch, y_dev_batch)
						total_dev_correct += num_dev_correct

					accuracy = float(total_dev_correct) / dev_size
					logging.critical('total_dev_correct: {}'.format(total_dev_correct))
					logging.critical('accuracy on dev: {}'.format(accuracy))

					if accuracy >= best_accuracy:
						best_accuracy = accuracy
						best_at_step = current_step
						path = saver.save(sess, checkpoint_prefix, global_step=current_step)
						logging.critical('Save the best model checkpoint to {} at evaluate step {}'.format(path, best_at_step))
						logging.critical('Best accuracy on dev set: {}, at step {}'.format(best_accuracy, best_at_step))

			logging.critical('Training is complete, testing the best model on test set')

			# Evaluate on test set (batch by batch) when training is complete
			saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))

			test_batches = data_helpers.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
			total_test_correct, predicted_labels = 0, []

			for test_batch in test_batches:
				x_test_batch, y_test_batch = zip(*test_batch)
				acc, loss, num_test_correct, predictions = dev_step(x_test_batch, y_test_batch)
				total_test_correct += int(num_test_correct)
				for prediction in predictions:
					predicted_labels.append(labels[prediction])

			df_test['PREDICTED'] = predicted_labels
			df_test_correct = df_test[df_test['PREDICTED'] == df_test['PROPOSED_CATEGORY']]
			df_test_non_correct = df_test[df_test['PREDICTED'] != df_test['PROPOSED_CATEGORY']]

			columns = sorted(df_test.columns, reverse=True)
			df_test.to_csv(trained_dir + 'predictions_all.csv', index=False, columns=columns, sep='|')
			df_test_correct.to_csv(trained_dir + 'predictions_correct.csv', index=False, columns=columns, sep='|')
			df_test_non_correct.to_csv(trained_dir + 'predictions_non_correct.csv', index=False, columns=columns, sep='|')

			logging.critical('Accuray on test set is {}'.format(float(total_test_correct) / test_size))
			logging.critical('total_test_correct: {}'.format(total_test_correct))

	with open(trained_dir + 'words_index.json', 'w') as outfile:
		json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
	with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
		pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
	with open(trained_dir + 'labels.json', 'w') as outfile:
		json.dump(labels, outfile, indent=4, ensure_ascii=False)

	os.rename(path, trained_dir + 'best_model.ckpt')
	os.rename(path + '.meta', trained_dir + 'best_model.meta')
	shutil.rmtree(checkpoint_dir)
	logging.critical('{} has been removed'.format(checkpoint_dir))

	params['sequence_length'] = x_.shape[1]

	# Save the trained parameters such as sequence length, needed for predicting unseen data in future
	with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
		json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)

if __name__ == '__main__':
	input_file = './data/bank_debit/input_40000.csv'
	train_cnn_lstm(input_file)

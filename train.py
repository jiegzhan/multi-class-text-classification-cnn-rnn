import argparse
import json
import logging
import os
import shutil
import time

import data_helper
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from text_cnn_rnn import TextCNNRNNModel

logging.getLogger().setLevel(logging.INFO)


def parse_args():
	parser = argparse.ArgumentParser(description="Train a TextCNN+GRU classifier.")
	parser.add_argument("data_file", help="Path to zip-compressed training CSV")
	parser.add_argument("config_file", help="Path to training config JSON")
	return parser.parse_args()


def load_config(config_file):
	with open(config_file, encoding="utf-8") as config_handle:
		return json.load(config_handle)


def build_model(params, vocab_size, num_classes, embedding_matrix):
	filter_sizes = [int(size) for size in params["filter_sizes"].split(",")]
	return TextCNNRNNModel(
		vocab_size=vocab_size,
		embedding_dim=params["embedding_dim"],
		non_static=params["non_static"],
		filter_sizes=filter_sizes,
		num_filters=params["num_filters"],
		max_pool_size=params["max_pool_size"],
		hidden_unit=params["hidden_unit"],
		num_classes=num_classes,
		embedding_matrix=embedding_matrix,
		l2_reg_lambda=params["l2_reg_lambda"],
		dropout_keep_prob=params["dropout_keep_prob"],
	)


def train_cnn_rnn():
	args = parse_args()
	if not os.path.isfile(args.data_file):
		raise FileNotFoundError(f"Training data not found: {args.data_file}")
	if not os.path.isfile(args.config_file):
		raise FileNotFoundError(f"Config file not found: {args.config_file}")

	params = load_config(args.config_file)
	seed = params.get("random_seed", 42)
	data_helper.set_seed(seed)

	x_, y_, vocabulary, vocabulary_inv, _, labels = data_helper.load_data(args.data_file)
	label_indices = data_helper.stratified_label_indices(y_)

	embedding_matrix = data_helper.load_random_embeddings(
		vocabulary_inv,
		params["embedding_dim"],
	)

	x_holdout, x_test, y_holdout, y_test, idx_holdout, idx_test = train_test_split(
		x_, y_, label_indices, test_size=0.1, random_state=seed, stratify=label_indices
	)
	x_train, x_dev, y_train, y_dev = train_test_split(
		x_holdout, y_holdout, test_size=0.1, random_state=seed, stratify=idx_holdout
	)

	logging.info("x_train: %s, x_dev: %s, x_test: %s", len(x_train), len(x_dev), len(x_test))

	timestamp = str(int(time.time()))
	trained_dir = f"./trained_results_{timestamp}/"
	if os.path.exists(trained_dir):
		shutil.rmtree(trained_dir)
	os.makedirs(trained_dir)

	model = build_model(params, len(vocabulary_inv), len(labels), embedding_matrix)
	model.compile(
		optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3, rho=0.9),
		loss="categorical_crossentropy",
		metrics=["accuracy"],
	)

	checkpoint_path = os.path.join(trained_dir, "best_model.keras")
	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(
			checkpoint_path,
			monitor="val_accuracy",
			save_best_only=True,
			mode="max",
		),
	]

	model.fit(
		x_train,
		y_train,
		batch_size=params["batch_size"],
		epochs=params["num_epochs"],
		validation_data=(x_dev, y_dev),
		callbacks=callbacks,
		verbose=1,
	)

	best_model = tf.keras.models.load_model(
		checkpoint_path,
		custom_objects={"TextCNNRNNModel": TextCNNRNNModel},
	)
	saved_model_dir = os.path.join(trained_dir, "saved_model")
	warmup_input = np.zeros((1, x_train.shape[1]), dtype=np.int32)
	best_model(warmup_input, training=False)
	best_model.export(saved_model_dir)

	test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
	logging.info("Test loss: %.4f, test accuracy: %.4f", test_loss, test_accuracy)

	params["sequence_length"] = int(x_train.shape[1])
	params["vocab_size"] = len(vocabulary_inv)
	params["num_classes"] = len(labels)

	with open(os.path.join(trained_dir, "words_index.json"), "w", encoding="utf-8") as outfile:
		json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
	with open(os.path.join(trained_dir, "labels.json"), "w", encoding="utf-8") as outfile:
		json.dump(labels, outfile, indent=4, ensure_ascii=False)
	with open(os.path.join(trained_dir, "trained_parameters.json"), "w", encoding="utf-8") as outfile:
		json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)

	logging.info("Training complete. Artifacts saved to %s", trained_dir)


if __name__ == "__main__":
	train_cnn_rnn()

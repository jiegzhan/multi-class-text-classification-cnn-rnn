import argparse
import json
import logging
import os
import shutil

import data_helper
import numpy as np
import tensorflow as tf
from text_cnn_rnn import TextCNNRNNModel

logging.getLogger().setLevel(logging.INFO)


def parse_args():
	parser = argparse.ArgumentParser(description="Predict crime categories for new descriptions.")
	parser.add_argument("trained_dir", help="Directory containing saved model artifacts")
	parser.add_argument("test_file", help="Pipe-delimited CSV with a Descript column")
	return parser.parse_args()


def normalize_trained_dir(trained_dir):
	if not trained_dir.endswith("/"):
		trained_dir += "/"
	return trained_dir


def load_trained_params(trained_dir):
	with open(trained_dir + "trained_parameters.json", encoding="utf-8") as config_handle:
		params = json.load(config_handle)
	with open(trained_dir + "words_index.json", encoding="utf-8") as vocab_handle:
		vocabulary = json.load(vocab_handle)
	with open(trained_dir + "labels.json", encoding="utf-8") as labels_handle:
		labels = json.load(labels_handle)
	return params, vocabulary, labels


def load_model(trained_dir):
	checkpoint_path = os.path.join(trained_dir, "best_model.keras")
	if os.path.isfile(checkpoint_path):
		return tf.keras.models.load_model(
			checkpoint_path,
			custom_objects={"TextCNNRNNModel": TextCNNRNNModel},
		)

	saved_model_dir = os.path.join(trained_dir, "saved_model")
	if os.path.isdir(saved_model_dir):
		return tf.keras.models.load_model(
			saved_model_dir,
			custom_objects={"TextCNNRNNModel": TextCNNRNNModel},
		)

	raise FileNotFoundError(
		f"No saved model found in {trained_dir}. Expected best_model.keras or saved_model/"
	)


def predict_unseen_data():
	args = parse_args()
	trained_dir = normalize_trained_dir(args.trained_dir)

	if not os.path.isdir(trained_dir):
		raise FileNotFoundError(f"Trained directory not found: {trained_dir}")
	if not os.path.isfile(args.test_file):
		raise FileNotFoundError(f"Test file not found: {args.test_file}")

	params, vocabulary, labels = load_trained_params(trained_dir)
	model = load_model(trained_dir)

	test_examples, y_true, df = data_helper.load_test_data(args.test_file, labels)
	x_test = data_helper.prepare_test_features(
		test_examples,
		vocabulary,
		params["sequence_length"],
	)

	timestamp = trained_dir.rstrip("/").split("_")[-1]
	predicted_dir = f"./predicted_results_{timestamp}/"
	if os.path.exists(predicted_dir):
		shutil.rmtree(predicted_dir)
	os.makedirs(predicted_dir)

	probabilities = model.predict(x_test, batch_size=params["batch_size"], verbose=0)
	predictions = np.argmax(probabilities, axis=1)
	predict_labels = [labels[index] for index in predictions]

	df["NEW_PREDICTED"] = predict_labels
	columns = sorted(df.columns, reverse=True)
	df.to_csv(
		os.path.join(predicted_dir, "predictions_all.csv"),
		index=False,
		columns=columns,
		sep="|",
	)

	if y_true is not None:
		y_true = np.asarray(y_true)
		accuracy = float(np.mean(predictions == y_true))
		logging.info("Prediction accuracy: %.4f", accuracy)

	logging.info("Prediction complete. Output saved to %s", predicted_dir)


if __name__ == "__main__":
	predict_unseen_data()

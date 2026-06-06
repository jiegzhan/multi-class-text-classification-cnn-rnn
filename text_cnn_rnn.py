import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers


@tf.keras.utils.register_keras_serializable(package="TextCNNRNN")
class TextCNNRNNModel(tf.keras.Model):
	"""Hybrid TextCNN + GRU classifier."""

	def __init__(
		self,
		vocab_size,
		embedding_dim,
		non_static,
		filter_sizes,
		num_filters,
		max_pool_size,
		hidden_unit,
		num_classes,
		embedding_matrix=None,
		l2_reg_lambda=0.0,
		dropout_keep_prob=0.5,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.non_static = non_static
		self.filter_sizes = list(filter_sizes)
		self.num_filters = num_filters
		self.max_pool_size = max_pool_size
		self.hidden_unit = hidden_unit
		self.num_classes = num_classes
		self.l2_reg_lambda = l2_reg_lambda
		self.dropout_keep_prob = dropout_keep_prob

		l2 = regularizers.l2(l2_reg_lambda) if l2_reg_lambda > 0 else None

		if embedding_matrix is None:
			embeddings_initializer = "uniform"
		else:
			embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix)

		self.embedding = tf.keras.layers.Embedding(
			input_dim=vocab_size,
			output_dim=embedding_dim,
			embeddings_initializer=embeddings_initializer,
			trainable=non_static,
			name="embedding",
		)

		self.conv_layers = []
		self.pool_layers = []
		for filter_size in self.filter_sizes:
			self.conv_layers.append(
				tf.keras.layers.Conv1D(
					filters=num_filters,
					kernel_size=filter_size,
					activation="relu",
					padding="same",
					kernel_regularizer=l2,
					name=f"conv_{filter_size}",
				)
			)
			self.pool_layers.append(
				tf.keras.layers.MaxPooling1D(
					pool_size=max_pool_size,
					strides=max_pool_size,
					padding="same",
					name=f"pool_{filter_size}",
				)
			)

		self.dropout = tf.keras.layers.Dropout(rate=1.0 - dropout_keep_prob)
		self.gru = tf.keras.layers.GRU(
			units=hidden_unit,
			return_sequences=False,
			kernel_regularizer=l2,
			recurrent_regularizer=l2,
			name="gru",
		)
		self.classifier = tf.keras.layers.Dense(
			units=num_classes,
			activation="softmax",
			kernel_regularizer=l2,
			name="classifier",
		)

	def call(self, inputs, training=False):
		x = self.embedding(inputs)

		pooled_outputs = []
		for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
			conv_output = conv_layer(x)
			pooled_outputs.append(pool_layer(conv_output))

		merged = tf.concat(pooled_outputs, axis=-1)
		merged = self.dropout(merged, training=training)
		gru_output = self.gru(merged)
		return self.classifier(gru_output)

	def get_config(self):
		config = super().get_config()
		config.update(
			{
				"vocab_size": self.vocab_size,
				"embedding_dim": self.embedding_dim,
				"non_static": self.non_static,
				"filter_sizes": self.filter_sizes,
				"num_filters": self.num_filters,
				"max_pool_size": self.max_pool_size,
				"hidden_unit": self.hidden_unit,
				"num_classes": self.num_classes,
				"l2_reg_lambda": self.l2_reg_lambda,
				"dropout_keep_prob": self.dropout_keep_prob,
			}
		)
		return config

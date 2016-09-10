"""
Optimized RNN implementations.
"""

import tensorflow as tf
import numpy as np
import tempfile

def make_example(sequence, labels):
	ex = tf.train.SequenceExample()
	sequence_length = len(sequence)
	ex.context.feature["length"].int64_list.value.append(sequence_length)

	fl_tokens = ex.feature_lists.feature_list["tokens"]
	fl_labels = ex.feature_lists.feature_list["labels"]

	for token, label in zip(sequence, labels):
		fl_tokens.feature.add().int64_list.value.append(token)
		fl_labels.feature.add().int64_list.value.append(label)

	return ex

def read_sample(sequence, label_sequence):
	ex = make_example(sequence, label_sequence).SerializeToString()

	# Non-sequential features
	context_features = {
		"length": tf.FixedLenFeature([], dtype=tf.int64)
	}

	# Sequential features
	sequence_features = {
		"tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
		"labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
	}

	context_parsed, sequence_parsed = tf.parse_single_sequence_example(
		serialized=ex,
		context_features=context_features,
		sequence_features=sequence_features)

	context = tf.contrib.learn.run_n(context_parsed, n=1, feed_dict=None)
	print(context[0])
	sequence = tf.contrib.learn.run_n(sequence_parsed, n=1, feed_dict=None)
	print(sequence[0])

def write_data(sequences, label_sequences):
	with tempfile.NamedTemporaryFile() as fp:
		writer = tf.python_io.TFRecordWriter(fp.name)
		for sequence, label_sequence in zip(sequences, label_sequences):
			ex = make_example(sequence, label_sequence)
			writer.write(ex.SerializeToString())
		writer.close()
		print("Wrote to {}".format(fp.name))

def pad_input(x):
	
	range_q = tf.train.range_input_producer(limit=5, shuffle=False)
	slice_end = range_q.dequeue()

	y = tf.slice(x, [0], [slice_end], name="y")

	batched_data = tf.train.batch(
		tensors=[y],
		batch_size=5,
		dynamic_pad=True,
		name="y_batch")

	res = tf.contrib.learn.run_n({"y": batched_data}, n=1, feed_dict=None)

	print "input_x:", x
	print("Batch shape: {}".format(res[0]["y"].shape))

	return res[0]["y"]

def dynamic_rnn_example():

	# Create input data
	X = np.random.randn(2, 10, 8)

	# The second example is of length 6 
	X[1,6:] = 0
	X_lengths = [10, 6]

	cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)

	outputs, last_states = tf.nn.dynamic_rnn(
		cell=cell,
		dtype=tf.float64,
		sequence_length=X_lengths,
		inputs=X)

	result = tf.contrib.learn.run_n(
	{"outputs": outputs, "last_states": last_states},
	n=1,
	feed_dict=None)

	assert result[0]["outputs"].shape == (2, 10, 64)
	print(result[0]["outputs"])

	# Outputs for the second example past past length 6 should be 0
	assert (result[0]["outputs"][1,6:,:] == np.zeros(cell.output_size)).all()

def dynamic_bidirectional_rnn():

	# Create input data
	X = np.random.randn(2, 10, 8)

	# The second example is of length 6 
	X[1,6:] = 0
	X_lengths = [10, 6]

	cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)

	outputs, states  = tf.nn.bidirectional_dynamic_rnn(
		cell_fw=cell,
		cell_bw=cell,
		dtype=tf.float64,
		sequence_length=X_lengths,
		inputs=X)

	output_fw, output_bw = outputs
	states_fw, states_bw = states

	result = tf.contrib.learn.run_n(
		{"output_fw": output_fw, "output_bw": output_bw, "states_fw": states_fw, "states_bw": states_bw},
		n=1,
		feed_dict=None)

	print(result[0]["output_fw"].shape)
	print(result[0]["output_bw"].shape)
	print(result[0]["states_fw"].h.shape)
	print(result[0]["states_bw"].h.shape)

def rnn_cell_wrappers_example():
	# Create input data
	X = np.random.randn(2, 10, 8)

	# The second example is of length 6 
	X[1,6,:] = 0
	X_lengths = [10, 6]

	cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)
	cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
	cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * 4, state_is_tuple=True)

	outputs, last_states = tf.nn.dynamic_rnn(
		cell=cell,
		dtype=tf.float64,
		sequence_length=X_lengths,
		inputs=X)

	result = tf.contrib.learn.run_n(
		{"outputs": outputs, "last_states": last_states},
		n=1,
		feed_dict=None)


	print(result[0]["outputs"].shape)
	print(result[0]["outputs"])
	assert result[0]["outputs"].shape == (2, 10, 64)

	# Outputs for the second example past past length 6 should be 0
	assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()

	print(result[0]["last_states"][0].h.shape)
	print(result[0]["last_states"][0].h)

def loss_masking_example():

	batch_size = 4 # sequences per batch
	max_seq_len = 8
	num_hidden_units = 128 # hidden units per state
	num_classes = 10

	# actual length of examples
	example_len = [1, 2, 3, 8]

	y = np.random.randint(1, 10, [batch_size, max_seq_len])
	for i, length in enumerate(example_len):
		y[i, length:] = 0

	rnn_outputs = tf.convert_to_tensor(np.random.randn(
		batch_size, max_seq_len, num_hidden_units),
		dtype=tf.float32)

	# W_softmax
	W = tf.get_variable(
		name="W",
		initializer=tf.random_normal_initializer(),
		shape=[num_hidden_units, num_classes])

	rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, num_hidden_units])
	logits_flat = tf.batch_matmul(rnn_outputs_flat, W)
	probs_flat = tf.nn.softmax(logits_flat)

	y_flat = tf.reshape(y, [-1])
	losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, y_flat)

	# apply mask on loss
	mask = tf.sign(tf.to_float(y_flat))
	masked_losses == mask * losses # 0 out losses for padding indexes

	# Reshape tp [batch_size, max_seq_len]
	masked_losses = tf.reshape(masked_losses, tf.shape(y))

	# Mean loss
	mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / example_len
	mean_loss = tf.reduce_mean(mean_loss_by_example)


if __name__ == '__main__':

	"""

	# Sample data
	sequences = [[1, 2, 3], [4, 5, 1], [1, 2]]
	label_sequences = [[0, 1, 0], [1, 0, 0], [1, 1]]

	# Exmaple on how to read the data
	read_sample(sequences[0], label_sequences[0])

	# Write data to a tmp file
	write_data(sequences, label_sequences)

	# Padding our inputs
	x = tf.range(1, 10, name="x")
	print pad_input(x)

	# Dynamic RNN
	dynamic_rnn_example()

	# Dynamic Bidrectional RNN
	dynamic_bidirectional_rnn()

	# RNN cell wrappers
	rnn_cell_wrappers_example()

	"""

	# RNN loss masking
	loss_masking_example()










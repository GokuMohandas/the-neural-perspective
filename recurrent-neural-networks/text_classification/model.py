import tensorflow as tf

def rnn_cell(FLAGS):

	# Get the cell type
	if FLAGS.rnn_unit == 'rnn':
		rnn_cell_type = tf.nn.rnn_cell.BasicRNNCell
	elif FLAGS.rnn_unit == 'gru':
		rnn_cell_type = tf.nn.rnn_cell.GRUCell
	elif FLAGS.rnn_unit == 'lstm':
		rnn_cell_type = tf.nn.rnn_cell.BasicLSTMCell
	else:
		raise Exception("Choose a valid RNN unit type.")

	# Single cell
	single_cell = rnn_cell_type(FLAGS.num_hidden_units, state_is_tuple=True)

	# Dropout
	single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=1-FLAGS.dropout)

	# Each state as one cell
	stacked_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * FLAGS.num_layers, state_is_tuple=True)

	return stacked_cell

def rnn_inputs(FLAGS, input_data):

	with tf.variable_scope('rnn_inputs', reuse=True):
		W_input = tf.get_variable("W_input", [FLAGS.en_vocab_size, FLAGS.num_hidden_units])

	# <num_examples, seq_len, num_hidden_units>
	embeddings = tf.nn.embedding_lookup(W_input, input_data)

	return embeddings

def rnn_softmax(FLAGS, outputs):
	with tf.variable_scope('rnn_softmax', reuse=True):
		W_softmax = tf.get_variable("W_softmax", [FLAGS.num_hidden_units, FLAGS.num_classes])
		b_softmax = tf.get_variable("b_softmax", [FLAGS.num_classes])

	logits = tf.matmul(outputs, W_softmax) + b_softmax

	return logits

def length(data):
	relevant = tf.sign(tf.abs(data))
	length = tf.reduce_sum(relevant, reduction_indices=1)
	length = tf.cast(length, tf.int32)
	return length

def last_relevant(output, length):
	batch_size = 64
	max_length = 61
	out_size = 300
	index = tf.range(0, batch_size) * max_length + (length - 1)

	flat = tf.reshape(output, [-1, out_size])
	relevant = tf.gather(flat, index)
	return index, relevant

def model(FLAGS, train=True, sample=False):

	# Placeholders
	inputs_X = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, None], name='inputs_X')
	targets_y = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.num_classes], name='targets_y')

	# RNN cell
	if not train:
		FLAGS.dropout=0.0
	stacked_cell = rnn_cell(FLAGS)

	# Inputs to RNN
	with tf.variable_scope('rnn_inputs'):
		W_input = tf.get_variable("W_input", [FLAGS.en_vocab_size, FLAGS.num_hidden_units])

	inputs = rnn_inputs(FLAGS, inputs_X)
	initial_state = stacked_cell.zero_state(FLAGS.batch_size, tf.float32)

	# Outputs from RNN
	seq_lens = length(inputs_X)
	all_outputs, state = tf.nn.dynamic_rnn(cell=stacked_cell, inputs=inputs, 
		initial_state=initial_state, sequence_length=seq_lens)

	if train:

		index, outputs = last_relevant(all_outputs, seq_lens)

		# Process RNN outputs
		with tf.variable_scope('rnn_softmax'):
			W_softmax = tf.get_variable("W_softmax", [FLAGS.num_hidden_units, FLAGS.num_classes])
			b_softmax = tf.get_variable("b_softmax", [FLAGS.num_classes])

		# Logits	
		logits = rnn_softmax(FLAGS, outputs)
		probabilities = tf.nn.softmax(logits)
		accuracy = tf.equal(tf.argmax(targets_y,1), tf.argmax(logits,1))

		# Loss
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, targets_y))
		final_state = state

		# Optimization
		lr = tf.Variable(0.0, trainable=False)
		trainable_vars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_vars),
										  FLAGS.max_gradient_norm) # clip the gradient to avoid vanishing or blowing up gradients
		optimizer = tf.train.AdamOptimizer(lr)
		train_optimizer = optimizer.apply_gradients(zip(grads, trainable_vars))

		return dict(inputs_X=inputs_X, targets_y=targets_y, seq_lens=seq_lens, index=index, all_outputs=all_outputs,
			lr=lr, outputs=outputs, logits=logits, accuracy=accuracy, loss=loss, train_optimizer=train_optimizer)

	elif sample:

		outputs = all_outputs[0] # this is taking all the ouputs for the first input sequence (only 1 input sequence since we are sampling)

		# Process RNN outputs
		with tf.variable_scope('rnn_softmax'):
			W_softmax = tf.get_variable("W_softmax", [FLAGS.num_hidden_units, FLAGS.num_classes])
			b_softmax = tf.get_variable("b_softmax", [FLAGS.num_classes])

		# Logits	
		logits = rnn_softmax(FLAGS, outputs)
		probabilities = tf.nn.softmax(logits)

		return dict(inputs_X=inputs_X, probabilities=probabilities)


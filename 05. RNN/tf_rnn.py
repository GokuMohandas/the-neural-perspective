import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, seq2seq
import sys

class parameters():

	def __init__(self):

		self.DATA_FILE = 'data/shakespeare.txt'
		self.CKPT_DIR = 'char_RNN_ckpt_dir'
		self.encoding = 'utf-8'
		self.SAVE_EVERY = 1 # save model every epoch
		self.TRAIN_RATIO = 0.8
		self.VALID_RATIO = 0.1

		self.NUM_EPOCHS = 100
		self.NUM_BATCHES = 50
		self.SEQ_LEN = 50
		self.MODEL = 'rnn'
		self.NUM_HIDDEN_UNITS = 128
		self.NUM_LAYERS = 2
		self.GRAD_CLIP = 5.0
		self.LEARNING_RATE = 0.002
		self.DECAY_RATE = 0.97
		self.DROPOUT = 0.5

		self.SAMPLE_LEN = 500
		self.START_TOKENS = "Thou "
		self.SAMPLE_TYPE = 1 # 0=argmax, 1=temperature based
		self.TEMPERATURE = 0.04

def generate_data(config):

	data = open(config.DATA_FILE, "r").read()
	chars = list(set(data))
	char_to_idx = {char:i for i, char in enumerate(chars)}
	idx_to_char = {i:char for i, char in enumerate(chars)}

	config.DATA_SIZE = len(data)
	config.NUM_CLASSES = len(chars)
	config.char_to_idx = char_to_idx
	config.idx_to_char = idx_to_char

	X = [config.char_to_idx[char] for char in data]
	y = X[1:]
	y[-1] = X[0]

	# Split into train, valid and test sets
	train_last_index = int(config.DATA_SIZE*config.TRAIN_RATIO)
	valid_first_index = train_last_index + 1
	valid_last_index = valid_first_index + int(config.DATA_SIZE*config.VALID_RATIO)
	test_first_index = valid_last_index + 1

	config.train_X = X[:train_last_index]
	config.train_y = y[:train_last_index]
	config.valid_X = X[valid_first_index:valid_last_index]
	config.valid_y = y[valid_first_index:valid_last_index]
	config.test_X = X[test_first_index:]
	config.test_y = y[test_first_index:]

	return config

def generate_batch(config, raw_X, raw_y):

	# Create batches from raw data
	batch_size = len(raw_X) // config.NUM_BATCHES # tokens per batch
	data_X = np.zeros([config.NUM_BATCHES, batch_size], dtype=np.int32)
	data_y = np.zeros([config.NUM_BATCHES, batch_size], dtype=np.int32)
	for i in range(config.NUM_BATCHES):
		data_X[i, :] = raw_X[batch_size * i: batch_size * (i+1)]
		data_y[i, :] = raw_y[batch_size * i: batch_size * (i+1)]

	# Even though we have tokens per batch,
	# We only want to feed in <SEQ_LEN> tokens at a time
	feed_size = batch_size // config.SEQ_LEN
	for i in range(feed_size):
		X = data_X[:, i * config.SEQ_LEN:(i+1) * config.SEQ_LEN]
		y = data_y[:, i * config.SEQ_LEN:(i+1) * config.SEQ_LEN]
		yield (X, y)

def generate_epochs(config):

	for i in range(config.NUM_EPOCHS):
		yield generate_batch(config, generate_data(config))

def model(config):

	if config.MODEL == 'rnn':
		rnn_cell_type = rnn_cell.BasicRNNCell
	elif config.MODEL == 'gru':
		rnn_cell_type = rnn_cell.GRUCell
	elif config.MODEL == 'lstm':
		rnn_cell_type = rnn_cell.BasicLSTMCell
	else:
		raise Exception("Choose a valid RNN unit type.")

	# Single cell
	single_cell = rnn_cell_type(config.NUM_HIDDEN_UNITS)
	# Dropout
	single_cell = rnn_cell.DropoutWrapper(single_cell, output_keep_prob=1-config.DROPOUT)
	# Each state as one cell
	cell = rnn_cell.MultiRNNCell([single_cell] * config.NUM_LAYERS)

	X = tf.placeholder(tf.int32, [config.NUM_BATCHES, config.SEQ_LEN])
	y = tf.placeholder(tf.int32, [config.NUM_BATCHES, config.SEQ_LEN])
	initial_state = cell.zero_state(config.NUM_BATCHES, tf.float32)

	with tf.variable_scope('rnn'):
		W_softmax = tf.get_variable("W_softmax", [config.NUM_HIDDEN_UNITS, config.NUM_CLASSES])
		b_softmax = tf.get_variable("b_softmax", [config.NUM_CLASSES])
		with tf.device("/cpu:0"):
			# Embedding is same as W_input
			embedding = tf.get_variable("embedding", [config.NUM_CLASSES, config.NUM_HIDDEN_UNITS])
			# <num_batches, seq_len, num_classes>
			inputs = tf.split(1, config.SEQ_LEN, 
						   tf.nn.embedding_lookup(embedding, X))
			# <seq_len, num_batches, num_classes> which is how we need for RNN input
			inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

	# <seq_len, num_batches, num_classes>
	outputs, final_state = seq2seq.rnn_decoder(inputs, initial_state, cell, scope='rnn')
	# <num_batches, num_hidden_units*seq_len> 
	output = tf.concat(1, outputs)
	# <num*batches*seq*len, num_hidden_units>
	output = tf.reshape(output, [-1, config.NUM_HIDDEN_UNITS])

	logits = tf.matmul(output, W_softmax) + b_softmax
	probabilities = tf.nn.softmax(logits)

	loss = seq2seq.sequence_loss_by_example([logits],
									[tf.reshape(y, [-1])],
									[tf.ones([config.NUM_BATCHES *
										     config.SEQ_LEN])],
									config.NUM_CLASSES)
	total_loss = tf.reduce_mean(loss)

	lr = tf.Variable(0.0, trainable=False)
	trainable_vars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, trainable_vars),
							    config.GRAD_CLIP)
	optimizer = tf.train.AdamOptimizer(lr)
	train_optimizer = optimizer.apply_gradients(zip(grads, trainable_vars))

	return dict(X=X, y=y, lr=lr, initial_state=initial_state,
		       total_loss=total_loss, train_optimizer=train_optimizer)

def train(config, g):

	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		print('All variables initialized')
		print("Training...\n")

		for epoch_num in range(config.NUM_EPOCHS):
			sess.run(tf.assign(g['lr'], config.LEARNING_RATE*(config.DECAY_RATE**epoch_num)))
			training_loss = []

			epoch_training_loss = 0.0
			num_feeds = (len(config.train_X) // config.NUM_BATCHES) // config.SEQ_LEN
			for batch_num, (train_X, train_y) in enumerate(generate_batch(config, config.train_X, config.train_y)):
				feed = {g['X']:train_X, g['y']: train_y}
				total_loss, _ = sess.run([g['total_loss'], g['train_optimizer']], feed)
				epoch_training_loss += total_loss

				sys.stdout.write("Training progress: %d%%   \r" % (100*batch_num/float(num_feeds)) )
				sys.stdout.flush()

			training_loss.append(epoch_training_loss/float(num_feeds))
			print "Avg loss per batch:", training_loss[-1]

if __name__ == '__main__':
	config = parameters()
	config = generate_data(config)
	g = model(config)
	train(config, g)












import numpy as np
import tensorflow as tf

class parameters():

	def __init__(self):
		self.DATA_SIZE = 10000
		self.DIMENSIONS = 2
		self.NUM_CLASSES = 3
		self.NUM_HIDDEN_UNITS = 100
		self.LEARNING_RATE = 1e-0
		self.REG = 1e-3
		self.NUM_EPOCHS = 100
		self.DISPLAY_STEP = 10

def load_data(config):

	np.random.seed(0)
	N = config.DATA_SIZE
	D = config.DIMENSIONS
	C = config.NUM_CLASSES 

	# Make synthetic spiral data
	X_original = np.zeros((N*C, D))
	y = np.zeros(N*C, dtype='uint8')
	for j in xrange(C):
		ix = range(N*j,N*(j+1))
		r = np.linspace(0.0,1,N) # radius
		t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
		X_original[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
		y[ix] = j

	X = np.hstack([X_original, np.ones((X_original.shape[0], 1))])
	D = X.shape[1]
	config.DIMENSIONS = D 

	print "X:", (np.shape(X)) # (300, 3)
	print "y:", (np.shape(y)) # (300,)

	return config, X, y

def linear_model(config, X, y):
	# Initialize weights
	W = 0.01 * np.random.randn(config.DIMENSIONS, config.NUM_CLASSES)

	for epoch_num in xrange(config.NUM_EPOCHS):

		# Class scores [NXC]
		logits = np.dot(X, W)

		# Class probabilities
		exp_logits = np.exp(logits)
		probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

		# Loss
		correct_class_logprobs = -np.log(probs[range(len(probs)), y])
		loss = np.sum(correct_class_logprobs) / config.DATA_SIZE
		loss += 0.5 * config.REG * np.sum(W*W)

		# show progress
		if epoch_num%config.DISPLAY_STEP == 0:
			print "Epoch: %i, loss: %.3f" % (epoch_num, loss)

		# Backpropagation
		dscores = probs
		dscores[range(len(probs)), y] -= 1
		dscores /= config.DATA_SIZE

		dW = np.dot(X.T, dscores)
		dW += config.REG*W

		W += -config.LEARNING_RATE * dW

	return W

def two_layer_NN(config, X, y):
	# Initialize weights
	W_1 = 0.01 * np.random.randn(config.DIMENSIONS, config.NUM_HIDDEN_UNITS)
	W_2 = 0.01 * np.random.randn(config.NUM_HIDDEN_UNITS, config.NUM_CLASSES)

	for epoch_num in xrange(config.NUM_EPOCHS):

		# Class scores [NXC]
		z_2 = np.dot(X, W_1)
		a_2 = np.maximum(0, z_2) # ReLU
		logits = np.dot(a_2, W_2)

		# Class probabilities
		exp_logits = np.exp(logits)
		probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

		# Loss
		correct_class_logprobs = -np.log(probs[range(len(probs)), y])
		loss = np.sum(correct_class_logprobs) / config.DATA_SIZE
		loss += 0.5 * config.REG * np.sum(W_1*W_1)
		loss += 0.5 * config.REG * np.sum(W_2*W_2)

		# show progress
		if epoch_num%config.DISPLAY_STEP == 0:
			print "Epoch: %i, loss: %.3f" % (epoch_num, loss)

		# Backpropagation
		dscores = probs
		dscores[range(len(probs)), y] -= 1
		dscores /= config.DATA_SIZE

		dW2 = np.dot(a_2.T, dscores) 
		dhidden = np.dot(dscores, W_2.T)
		dhidden[a_2 <= 0] = 0 # ReLu backprop
		dW1 = np.dot(X.T, dhidden)

		dW2 += config.REG * W_2
		dW1 += config.REG * W_1

		W_1 += -config.LEARNING_RATE * dW1
		W_2 += -config.LEARNING_RATE * dW2

	return W_1, W_2


def accuracy(X, y, W_1, W_2=None):
	logits = np.dot(X, W_1)
	if W_2 is None:
		predicted_class = np.argmax(logits, axis=1)
		print "Accuracy: %.3f" % (np.mean(predicted_class == y))
	else:
		z_2 = np.dot(X, W_1)
		a_2 = np.maximum(0, z_2)
		logits = np.dot(a_2, W_2)
		predicted_class = np.argmax(logits, axis=1)
		print "Accuracy: %.3f" % (np.mean(predicted_class == y))

def tf_NN_model(config):

	# Placeholders
	X = tf.placeholder("float", [None, config.DIMENSIONS])
	y = tf.placeholder("float", [None, config.NUM_CLASSES])

	# Weights
	W1 = tf.Variable(tf.random_normal([config.DIMENSIONS, config.NUM_HIDDEN_UNITS], stddev=0.01), "W1")
	W2 = tf.Variable(tf.random_normal([config.NUM_HIDDEN_UNITS, config.NUM_CLASSES], stddev=0.01), "W2")

	with tf.name_scope('forward_pass') as scope:
		z_2 = tf.matmul(X, W1)
		a_2 = tf.nn.relu(z_2)
		logits = tf.matmul(a_2, W2)

	# Add summary ops to collect data
	W_1 = tf.histogram_summary("W1", W1)
	W_2 = tf.histogram_summary("W2", W2)

	with tf.name_scope('cost') as scope:
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y)) + 0.5 * config.REG * tf.reduce_sum(W1*W1) + 0.5 * config.REG * tf.reduce_sum(W2*W2)
		tf.scalar_summary("cost", cost)

	with tf.name_scope('train') as scope:
		optimizer = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE).minimize(cost)

	return dict(X=X, y=y, W1=W1, W2=W2, logits=logits, cost=cost, optimizer=optimizer)

def train(config, g, X, y):

	# Merge all summaries into a single operator
	merged_summary_op = tf.merge_all_summaries()

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		summary_writer = tf.train.SummaryWriter('logs', graph=sess.graph)

		# y to categorical
		Y = tf.one_hot(y, 3).eval()

		for epoch_num in range(config.NUM_EPOCHS):
			logits, training_loss, _ = sess.run([g['logits'], 
										  g['cost'], 
										  g['optimizer']],
										  feed_dict={g['X']:X, g['y']:Y})
			# Display
			if epoch_num%config.DISPLAY_STEP == 0:
				print "EPOCH %i: \n Training loss: %.3f, Accuracy: %.3f" % (
					epoch_num, training_loss, np.mean(np.argmax(logits, 1) == y))

				# Write logs for each epoch_num
				summary_str = sess.run(merged_summary_op, feed_dict={g['X']:X, g['y']:Y})
				summary_writer.add_summary(summary_str, epoch_num)

if __name__ == '__main__':
	config = parameters()
	config, X, y = load_data(config)
	g = tf_NN_model(config)
	train(config, g, X, y)


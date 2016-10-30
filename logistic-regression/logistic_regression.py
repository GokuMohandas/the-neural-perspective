import tensorflow as tf
import numpy as np
import input_data

class parameters():

	def __init__(self):
		self.LEARNING_RATE = 0.05
		self.NUM_EPOCHS = 500
		self.DISPLAY_STEP = 1 # epoch

def load_data():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	trainX, trainY, testX, testY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	return trainX, trainY, testX, testY

def create_model(sess, learning_rate):
	tf_model = model(learning_rate)
	sess.run(tf.initialize_all_variables())
	return tf_model

class model(object):

	def __init__(self, learning_rate):

		# Placeholders
		self.X = tf.placeholder("float", [None, 784])
		self.y = tf.placeholder("float", [None, 10])

		# Weights
		with tf.variable_scope('weights'):
			W = tf.Variable(tf.random_normal([784, 10], stddev=0.01), "W")

		self.logits = tf.matmul(self.X, W)
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y))
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)

		# Prediction
		self.prediction = tf.argmax(self.logits, 1)

	def step(self, sess, batch_X, batch_y, forward_only=True):

		input_feed = {self.X: batch_X, self.y: batch_y}
		if not forward_only:
			output_feed = [self.prediction,
						self.cost,
						self.optimizer]
		else:
			output_feed = [self.cost]

		outputs = sess.run(output_feed, input_feed)

		if not forward_only:
			return outputs[0], outputs[1], outputs[2]
		else:
			return outputs[0]

def train(FLAGS):

	with tf.Session() as sess:

		model = create_model(sess, FLAGS.LEARNING_RATE)
		trainX, trainY, testX, testY = load_data()

		for epoch_num in range(FLAGS.NUM_EPOCHS):
			prediction, training_loss, _ = model.step(sess, trainX, trainY, forward_only=False)

			# Display
			if epoch_num%FLAGS.DISPLAY_STEP == 0:
				print "EPOCH %i: \n Training loss: %.3f, Test loss: %.3f" % (
					epoch_num, training_loss, model.step(sess, testX, testY, forward_only=True))

if __name__== '__main__':
	FLAGS = parameters()
	train(FLAGS)


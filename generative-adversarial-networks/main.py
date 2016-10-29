# Modified from https://github.com/ericjang/genadv_tutorial/blob/master/genadv1.ipynb

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for pretty plots
from scipy.stats import norm

class parameters():

	def __init__(self):

		self.NUM_EPOCHS = 10000
		self.BATCH_SIZE = 200 # samples per minibatch

class Data_Distribution(object):

	def __init__(self):
		self.mu = 0
		self.sigma = 1

def mlp(inputs, output_dim):

	W1 = tf.get_variable("W1", 
		shape=[inputs.get_shape()[1], 10], 
		initializer=tf.random_normal_initializer())
	b1 = tf.get_variable("b1", 
		shape=[10], 
		initializer=tf.constant_initializer(0.0))
	W2 = tf.get_variable("W2", 
		shape=[10, 10], 
		initializer=tf.random_normal_initializer())
	b2 = tf.get_variable("b2", 
		shape=[10], 
		initializer=tf.constant_initializer(0.0))
	W3 = tf.get_variable("W3", 
		shape=[10, output_dim], 
		initializer=tf.random_normal_initializer())
	b3 = tf.get_variable("b3", 
		shape=[output_dim], 
		initializer=tf.constant_initializer(0.0))

	fc1 = tf.nn.tanh(tf.matmul(inputs, W1) + b1)
	fc2 = tf.nn.tanh(tf.matmul(fc1, W2) + b2)
	fc3 = tf.nn.tanh(tf.matmul(fc2, W3) + b3)
	return fc3, [W1, b1, W2, b2, W3, b3]

def momentum_optimizer(FLAGS, loss,var_list):
	batch = tf.Variable(0)
	learning_rate = tf.train.exponential_decay(
		0.001, # starting learning rate
		batch, # index
		FLAGS.NUM_EPOCHS // 4, # num of times to decay
		0.95, # decay rate
		staircase=True)
	optimizer=tf.train.MomentumOptimizer(learning_rate,0.6).minimize(loss,global_step=batch,var_list=var_list)
	#optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
	return optimizer

def pretrained_discriminator(FLAGS):

	with tf.variable_scope("D_pretrain") as scope:
		inputs = tf.placeholder(tf.float32, 
			shape=(FLAGS.BATCH_SIZE, 1))
		labels = tf.placeholder(tf.float32,
			shape=(FLAGS.BATCH_SIZE, 1))
		D, theta_d = mlp(inputs, 1)
		loss_d = tf.reduce_mean(tf.square(D-labels))
		optimizer_d = momentum_optimizer(FLAGS, loss_d, None)

	return dict(inputs=inputs, labels=labels, D=D, theta_d=theta_d,
		loss_d=loss_d, optimizer_d=optimizer_d)

def GAN(FLAGS):

	with tf.variable_scope("G"):
		z = tf.placeholder(tf.float32, shape=(FLAGS.BATCH_SIZE, 1))
		G, theta_g = mlp(z, 1)
		G = tf.mul(5.0, G) # scale (originally everything is between -1 and 1 but we want -5 and 5)

	with tf.variable_scope("D_pretrain") as scope:

		scope.reuse_variables()

		# D for inputs from training data
		x = tf.placeholder(tf.float32, shape=(FLAGS.BATCH_SIZE, 1))
		fc, theta_d = mlp(x, 1)
		D1 = tf.maximum(tf.minimum(fc, 0.99), 0.01)

		# D for inputs from G
		fc, theta_d = mlp(G, 1)
		D2 = tf.maximum(tf.minimum(fc, 0.99), 0.01)

	objective_d = tf.reduce_mean(tf.log(D1) + tf.log(1-D2))
	objective_g = tf.reduce_mean(tf.log(D2)) # note, we want cost to be 1-objective_g, so we do log(D2) instead of log(1-D2)

	optimizer_d = momentum_optimizer(FLAGS, 1-objective_d, theta_d) # maximize objective_d so cost is 1-objective_d
	optimizer_g = momentum_optimizer(FLAGS, 1-objective_g, theta_g) 

	return dict(z=z, x=x,
		D1=D1, theta_d=theta_d, optimizer_d=optimizer_d, objective_d=objective_d, 
		G=G, theta_g=theta_g, optimizer_g=optimizer_g, objective_g=objective_g)

def plot_d(FLAGS, p_data, gan_model):

	f, ax = plt.subplots(1)
	r = 1000
	
	# p_data
	xs = np.linspace(-5, 5, r)
	ax.plot(xs, norm.pdf(xs, loc=p_data.mu, scale=p_data.sigma), label='p_data')

	# p_discriminator
	ds = np.zeros((r, 1))
	M = FLAGS.BATCH_SIZE
	for i in range(r/M):
		x = np.reshape(xs[M*i:M*(i+1)], (M,1))
		ds[M*i:M*(i+1)] = sess.run(gan_model['D'],
			feed_dict = {gan_model['inputs']:x})
	ax.plot(xs, ds, label='decision boundary')

	#ax.set_ylim(0,1.1)
	plt.legend()
	plt.show()

def plot_training_loss(FLAGS, p_data, gan_model):

	r = 1000

	# Training loss
	lh = np.zeros(r)
	for i in range(r):
		d = (np.random.random(FLAGS.BATCH_SIZE)-0.5) * 10.0 # lowest is -5 (0-0.5 * 10) to 5 (0+0.5 * 10)
		labels = norm.pdf(d, loc=p_data.mu, scale=p_data.sigma)
		lh[i], _= sess.run([gan_model['loss_d'], gan_model['optimizer_d']],
			feed_dict={gan_model['inputs']: np.reshape(d, (FLAGS.BATCH_SIZE, 1)),
					gan_model['labels']: np.reshape(labels, (FLAGS.BATCH_SIZE, 1))})

	plt.plot(lh)
	plt.show()

def plot_g(FLAGS, p_data, gan_model):

	f, ax = plt.subplots(1)
	r = 1000
	
	# p_data
	xs = np.linspace(-5, 5, r)
	ax.plot(xs, norm.pdf(xs, loc=p_data.mu, scale=p_data.sigma), 
		label='p_data')

	# p_discriminator
	ds = np.zeros((r, 1))
	M = FLAGS.BATCH_SIZE
	for i in range(r/M):
		x = np.reshape(xs[M*i:M*(i+1)], (M,1))
		ds[M*i:M*(i+1)] = sess.run(gan_model['D1'],
			feed_dict = {gan_model['x']:x})
	ax.plot(xs, ds, label='decision boundary')

	# p_g
	zs = np.linspace(-5, 5, r)
	gs = np.zeros((r, 1))
	for i in range(r/M):
		z = np.reshape(zs[M*i:M*(i+1)], (M, 1))
		gs[M*i:M*(i+1)] = sess.run(gan_model['G'], 
			feed_dict={gan_model['z']:z})
	histc, edges = np.histogram(gs, bins=10)
	ax.plot(np.linspace(-5, 5, 10), histc/float(r), label='p_g')

	#ax.set_ylim(0,1.1)
	plt.legend()
	plt.show()

if __name__ == '__main__':

	FLAGS = parameters()

	p_data = Data_Distribution()

	pretrained_discriminator_model = pretrained_discriminator(FLAGS)
	gan_model = GAN(FLAGS)

	with tf.Session() as sess:
		tf.initialize_all_variables().run()

		#plot_d(FLAGS, p_data, pretrained_discriminator_model)
		#plot_training_loss(FLAGS, p_data, pretrained_discriminator_model)
		#plot_d(FLAGS, p_data, pretrained_discriminator_model)

		#plot_g(FLAGS, p_data, gan_model)


		# Ian Goodfellow's GAN algorithm
		k = 1
		histd, histg = np.zeros(FLAGS.NUM_EPOCHS), np.zeros(FLAGS.NUM_EPOCHS)
		for i in range(FLAGS.NUM_EPOCHS):

			for j in range(k):
				x = np.random.normal(p_data.mu, p_data.sigma, FLAGS.BATCH_SIZE)
				x.sort()
				z = np.linspace(
					int(p_data.mu-4*p_data.sigma), 
					int(p_data.mu+4*p_data.sigma), 
					FLAGS.BATCH_SIZE) + np.random.random(FLAGS.BATCH_SIZE)*0.0
				histd[i], _ = sess.run([gan_model['objective_d'], gan_model['optimizer_d']],
					feed_dict={gan_model['x']: np.reshape(x, (FLAGS.BATCH_SIZE, 1)),
							 gan_model['z']: np.reshape(z, (FLAGS.BATCH_SIZE, 1))})

			# After k updates for discrimintor, we do one update for G
			z = np.linspace(
					int(p_data.mu-4*p_data.sigma), 
					int(p_data.mu+4*p_data.sigma), 
					FLAGS.BATCH_SIZE) + np.random.random(FLAGS.BATCH_SIZE)*0.01
			histg[i], _ = sess.run([gan_model['objective_g'], gan_model['optimizer_g']],
					feed_dict={gan_model['z']: np.reshape(z, (FLAGS.BATCH_SIZE, 1))})
			if i % (FLAGS.NUM_EPOCHS//10) == 0:
				print (float(i)/float(FLAGS.NUM_EPOCHS))


		plt.plot(range(FLAGS.NUM_EPOCHS),histd, label='obj_d')
		plt.plot(range(FLAGS.NUM_EPOCHS), 1-histg, label='obj_g')
		plt.legend()
		plt.show()

		plot_g(FLAGS, p_data, gan_model)







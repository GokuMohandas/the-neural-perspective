import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os

class parameters():

    """
    Hyperparameteres for the GAN
    """

    def __init__(self):

        # True data distribution
        self.mu = 0
        self.sigma = 1

        self.num_epochs = 10000
        self.num_points = 1000
        self.batch_size = 200
        self.x_dimensions = 1
        self.y_dimensions = 1
        self.num_hidden_units = 10

        self.ckpt_dir = "checkpoints"


def momentum_optimizer(num_epochs, loss, var_list):

    """
    Our momentum optimizer to use for training.
    """

    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.001, # starting learning rate
        batch, # index
        num_epochs // 4, # num of times to decay
        0.95, # decay rate
        staircase=True)
    optimizer=tf.train.MomentumOptimizer(learning_rate,0.6).minimize(
        loss, global_step=batch, var_list=var_list)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    return optimizer

def linear(inputs, output_dim, scope=None, stddev=1.0):
    """
    We will be using a lot of weights so let's make a function for
    it and the matmul operation.
    """
    with tf.variable_scope(scope or 'linear'):
        W = tf.get_variable("W",
                shape=[inputs.get_shape()[1], output_dim],
                initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b",
                shape=[output_dim],
                initializer=tf.constant_initializer(0.0))
    return tf.matmul(inputs, W) + b

def discriminator(inputs, weights_dim, use_minibatch):
    """
    """
    h0 = tf.tanh(linear(inputs, weights_dim * 2, 'd0'))
    h1 = tf.tanh(linear(h0, weights_dim * 2, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if use_minibatch:
        h2 = minibatch(h1)
    else:
        h2 = tf.tanh(linear(h1, weights_dim * 2, scope='d2'))

    h3 = tf.tanh(linear(h2, 1, scope='d3'))
    return h3, h2

def generator(inputs, weights_dim):
    h0 = tf.nn.softplus(linear(inputs, weights_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1

# https://github.com/AYLIEN/gan-intro/blob/master/gan.py
def minibatch(inputs, num_kernels=5, kernel_dim=3):
    x = linear(inputs, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(
        tf.transpose(activation, [1, 2, 0]), 0)
    eps = tf.expand_dims(np.eye(int(inputs.get_shape()[0]), dtype=np.float32), 1)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2) + eps
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat(1, [inputs, minibatch_features])


class pretrain_D_model(object):

    def __init__(self, FLAGS):
        # Pretrain D
        with tf.variable_scope("D_pretrain"):
            self.pretrained_inputs = tf.placeholder(tf.float32,
                shape=[FLAGS.batch_size, 1], name="pretrain_inputs")
            self.pretrain_labels = tf.placeholder(tf.float32,
                shape=[None, None], name="pretrain_labels")
            self.D, _ = discriminator(self.pretrained_inputs,
                                     FLAGS.num_hidden_units,
                                     use_minibatch=True)
            self.pretrain_loss = tf.reduce_mean(tf.square(
                self.D-self.pretrain_labels))

        vars_ = tf.trainable_variables()
        self.theta_D_pretrain = [
            v for v in vars_ if v.name.startswith('D_pretrain/')]

        self.optimizer_d = momentum_optimizer(FLAGS.num_epochs,
                                        self.pretrain_loss,
                                        None)

    def step_pretrain_D(self, sess, batch_X, batch_y):

        input_feed = {self.pretrained_inputs: batch_X,
                      self.pretrain_labels: batch_y}
        output_feed = [self.D, self.theta_D_pretrain,
                       self.pretrain_loss, self.optimizer_d]

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2], outputs[3]


class GAN_model(object):

    def __init__(self, FLAGS):


        # Pretraining the discriminator D with training data samples
        # to ensure that it can accurately predict that they do indeed
        # come from the training data.

        # PRETRAIN DISCRIMINATOR D
        # -------------------------------------------------------------

        with tf.variable_scope("G"):
            self.z = tf.placeholder(tf.float32,
                shape=[FLAGS.batch_size, 1], name='z')
            G = generator(self.z,
                            FLAGS.num_hidden_units)

            # We need to scale our X' to match X's scale
            # So G doesn't have to learn the scaling too
            # which may take more time and/or lead to never converging
            self.G = tf.mul(5.0, G)

        with tf.variable_scope("D") as scope:
            # Feed X into D
            self.X = tf.placeholder(tf.float32,
                shape=[FLAGS.batch_size, 1], name="X")
            D_X, self.fc_D = discriminator(self.X,
                                 FLAGS.num_hidden_units,
                                 use_minibatch=True)
            # Scale the prediction
            self.D_X = tf.maximum(tf.minimum(D_X, 0.99), 0.01)

            scope.reuse_variables()

            # Feed X' into D
            D_X_prime, self.fc_G = discriminator(self.G,
                                             FLAGS.num_hidden_units,
                                             use_minibatch=True)
            # Scale the prediction
            self.D_X_prime = tf.maximum(tf.minimum(D_X_prime, 0.99), 0.01)

        vars_ = tf.trainable_variables()
        self.theta_G = [v for v in vars_ if v.name.startswith('G/')]
        self.theta_D = [v for v in vars_ if v.name.startswith('D/')]

        # Objective functions
        self.objective_D = tf.reduce_mean(tf.log(self.D_X) + \
                           tf.log(1-self.D_X_prime))
        self.objective_G = tf.reduce_mean(tf.log(self.D_X_prime))

        # Train optimizers
        self.optimizer_D = momentum_optimizer(FLAGS.num_epochs,
                                         1-self.objective_D,
                                         self.theta_D)
        self.optimizer_G = momentum_optimizer(FLAGS.num_epochs,
                                         1-self.objective_G,
                                         self.theta_G)

    def step_D(self, sess, batch_z, batch_X):
        input_feed = {self.z: batch_z, self.X: batch_X}
        output_feed = [self.objective_D,
                       self.optimizer_D]

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def step_G(self, sess, batch_z, batch_X):
        input_feed = {self.z: batch_z, self.X: batch_X}
        output_feed = [self.objective_G,
                       self.optimizer_G]

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]

def plot_data_and_D(sess, model, FLAGS):

    # True data distribution with untrained D
    f, ax = plt.subplots(1)

    # p_data
    X = np.linspace(int(FLAGS.mu-3.0*FLAGS.sigma),
                    int(FLAGS.mu+3.0*FLAGS.sigma),
                    FLAGS.num_points)
    y = norm.pdf(X, loc=FLAGS.mu, scale=FLAGS.sigma)
    ax.plot(X, y, label='p_data')

    # Untrained p_discriminator
    untrained_D = np.zeros((FLAGS.num_points,1))
    for i in range(FLAGS.num_points/FLAGS.batch_size):
        batch_X = np.reshape(
            X[FLAGS.batch_size*i:FLAGS.batch_size*(i+1)],
            (FLAGS.batch_size,1))
        untrained_D[FLAGS.batch_size*i:FLAGS.batch_size*(i+1)] = \
            sess.run(model.D,
                feed_dict={model.pretrained_inputs: batch_X})
    ax.plot(X, untrained_D, label='untrained_D')

    plt.legend()
    plt.show()

def plot_G(sess, GAN, FLAGS, save=False, num_epoch=None):

    # True data distribution with untrained D
    f, ax = plt.subplots(1)

    # p_data
    X = np.linspace(int(FLAGS.mu-3.0*FLAGS.sigma),
                    int(FLAGS.mu+3.0*FLAGS.sigma),
                    FLAGS.num_points)
    y = norm.pdf(X, loc=FLAGS.mu, scale=FLAGS.sigma)
    ax.plot(X, y, label='p_data')

    # Untrained p_discriminator
    trained_D = np.zeros((FLAGS.num_points,1))
    for i in range(FLAGS.num_points/FLAGS.batch_size):
        batch_X = np.reshape(
            X[FLAGS.batch_size*i:FLAGS.batch_size*(i+1)],
            (FLAGS.batch_size,1))
        trained_D[FLAGS.batch_size*i:FLAGS.batch_size*(i+1)] = \
            sess.run(GAN.D_X,
                feed_dict={GAN.X: batch_X})
    ax.plot(X, trained_D, label='trained_D')

    # Plotting G
    X = np.linspace(int(FLAGS.mu-3.0*FLAGS.sigma),
                    int(FLAGS.mu+3.0*FLAGS.sigma),
                    FLAGS.num_points)
    z = np.linspace(int(FLAGS.mu-3.0*FLAGS.sigma),
                        int(FLAGS.mu+3.0*FLAGS.sigma),
                        FLAGS.num_points) + \
                        np.random.random(FLAGS.num_points)*0.01
    G = np.zeros((FLAGS.num_points, 1))
    for i in range(FLAGS.num_points/FLAGS.batch_size):
        batch_z = np.reshape(z[FLAGS.batch_size*i:FLAGS.batch_size*(i+1)],
                            (FLAGS.batch_size, 1))
        G[FLAGS.batch_size*i:FLAGS.batch_size*(i+1)] = \
            sess.run(GAN.G, feed_dict={GAN.z: batch_z})

    hist_G, edges = np.histogram(G, bins=10)
    ax.plot(np.linspace(int(FLAGS.mu-3.0*FLAGS.sigma),
                    int(FLAGS.mu+3.0*FLAGS.sigma), 10),
        hist_G/float(FLAGS.num_points), label='untrained_G')

    if not save:
        plt.legend()
        plt.show()
    elif save:
        f.savefig('plots/%i_epoch' % num_epoch)


def train(FLAGS):

    # Call both models
    pretrain_D = pretrain_D_model(FLAGS)
    GAN = GAN_model(FLAGS)

    with tf.Session() as sess:

        # Initialize all tf variables
        tf.initialize_all_variables().run()

        # Plot the untrained D
        plot_data_and_D(sess, pretrain_D, FLAGS)

        # Let's pretrain the discriminator D
        losses = np.zeros(FLAGS.num_points)
        for i in xrange(FLAGS.num_points):
            batch_X = (np.random.uniform(int(FLAGS.mu-3.0*FLAGS.sigma),
                                        int(FLAGS.mu+3.0*FLAGS.sigma),
                                        FLAGS.batch_size))
            batch_y = norm.pdf(batch_X, loc=FLAGS.mu, scale=FLAGS.sigma)
            batch_X = np.reshape(batch_X, (-1,1))
            batch_y = np.reshape(batch_y, (-1,1))
            D, theta_D, losses[i], _ = pretrain_D.step_pretrain_D(sess,
                                                                batch_X,
                                                                batch_y)

        # Plot the losses
        plt.plot(xrange(FLAGS.num_points), losses, label='pretraining_loss')
        plt.show()

        # Plot the trained D
        plot_data_and_D(sess, pretrain_D, FLAGS)

        # Initialize weights for D from pretrain_D
        for i,v in enumerate(GAN.theta_D):
            sess.run(v.assign(theta_D[i]))

        # Let's plot the untrained G
        plot_G(sess, GAN, FLAGS)

        # Let's train the GAN
        k=1
        objective_Ds = np.zeros(FLAGS.num_epochs)
        optimizer_Gs = np.zeros(FLAGS.num_epochs)
        for i in xrange(FLAGS.num_epochs):

            if i%1000 == 0:
                print i/float(FLAGS.num_epochs)

            # k updates to discriminator
            for j in xrange(k):
                batch_X = np.random.normal(FLAGS.mu,
                                           FLAGS.sigma,
                                           FLAGS.batch_size)
                batch_X.sort()
                batch_X = np.reshape(batch_X, (FLAGS.batch_size, 1))
                batch_z = np.linspace(int(FLAGS.mu-3.0*FLAGS.sigma),
                                      int(FLAGS.mu+3.0*FLAGS.sigma),
                                    FLAGS.batch_size) + \
                                    np.random.random(FLAGS.batch_size)*0.01
                batch_z = np.reshape(batch_z, (FLAGS.batch_size, 1))

                ''' Both batch_X and batch_Z are sorted for manifold alignment.
                Manifold alignment
                (https://sites.google.com/site/changwangnk/home/ma-html) will help
                G learn the underlying structure of p_data. By sorting, we are
                making this process easier, since now adjacent points in Z
                will directly map to adjacent points in X (and even the scale
                will match since we scaled outputs from G.)
                '''

                objective_Ds[i], _ = GAN.step_D(sess, batch_z, batch_X)

            # 1 update to G
            batch_z = np.linspace(int(FLAGS.mu-3.0*FLAGS.sigma),
                                  int(FLAGS.mu+3.0*FLAGS.sigma),
                                    FLAGS.batch_size) + \
                                    np.random.random(FLAGS.batch_size)*0.01
            batch_z = np.reshape(batch_z, (FLAGS.batch_size, 1))
            optimizer_Gs[i], _ = GAN.step_G(sess, batch_z, batch_X)

            #if i%100 == 0:
            #    plot_G(sess, GAN, FLAGS, save=True, num_epoch=i)

        # Plot objectives
        plt.plot(xrange(FLAGS.num_epochs), objective_Ds, label="objective_D")
        plt.plot(xrange(FLAGS.num_epochs), 1-optimizer_Gs, label="optimizer_G")

        plt.legend()
        plt.show()

        # Plot trained G
        plot_G(sess, GAN, FLAGS)



if __name__ == '__main__':

    FLAGS = parameters()
    train(FLAGS)










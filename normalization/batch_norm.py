import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import (
    input_data,
)

data = input_data.read_data_sets('MNIST_data', one_hot=True)

class parameters():

    def __init__(self):
        self.num_epochs = 10
        self.batch_size = 64

        self.epsilon = 1e-3
        self.learning_rate = 0.01

        self.record_every = 50


def train(FLAGS):
    """
    Train our MLP and MLP_BN.
    """

    # Intialize weights with numpy so we can pass same to both MLP and MLP_BN
    W1_init = np.random.normal(size=(784,100)).astype(np.float32)
    W2_init = np.random.normal(size=(100,100)).astype(np.float32)
    W3_init = np.random.normal(size=(100,10)).astype(np.float32)

    X = tf.placeholder(tf.float32, [None, None], 'X')
    y = tf.placeholder(tf.float32, [None, None], 'y')

    # MLP
    W1 = tf.Variable(W1_init)
    b1 = tf.Variable(tf.zeros([100]))
    z1 = tf.matmul(X,W1)+b1
    fc1 = tf.nn.relu(z1)

    W2 = tf.Variable(W2_init)
    b2 = tf.Variable(tf.zeros([100]))
    z2 = tf.matmul(fc1,W2)+b2
    fc2 = tf.nn.relu(z2)

    W3 = tf.Variable(W3_init)
    b3 = tf.Variable(tf.zeros([10]))
    fc3  = tf.matmul(fc2,W3)+b3

    mlp_probabilities = tf.nn.softmax(fc3)
    mlp_acc = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1)), tf.float32))
    mlp_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(fc3, y))
    mlp_optimizer = tf.train.AdamOptimizer(
        FLAGS.learning_rate).minimize(mlp_loss)

    # MLP with Batchnorm
    # Naive BN layer
    scale1 = tf.Variable(tf.ones([100]))
    shift1 = tf.Variable(tf.zeros([100]))
    W1_BN = tf.Variable(W1_init)
    b1_BN = tf.Variable(tf.zeros([100]))
    z1_BN = tf.matmul(X,W1_BN)+b1_BN
    mean1, var1 = tf.nn.moments(z1_BN, [0])
    BN1 = (z1_BN - mean1) / tf.sqrt(var1 + FLAGS.epsilon)
    BN1 = scale1*BN1 + shift1
    fc1_BN = tf.nn.relu(BN1)

    # TF BN layer
    scale2 = tf.Variable(tf.ones([100]))
    shift2 = tf.Variable(tf.zeros([100]))
    W2_BN = tf.Variable(W2_init)
    b2_BN = tf.Variable(tf.zeros([100]))
    z2_BN = tf.matmul(fc1_BN,W2_BN)+b2_BN
    mean2, var2 = tf.nn.moments(z2_BN, [0])
    BN2 = tf.nn.batch_normalization(z2_BN,mean2,var2,shift2,scale2,FLAGS.epsilon)
    fc2_BN = tf.nn.relu(BN2)

    W3_BN = tf.Variable(W3_init)
    b3_BN = tf.Variable(tf.zeros([10]))
    fc3_BN = tf.matmul(fc2_BN,W3_BN)+b3_BN

    mlp_BN_probabilities = tf.nn.softmax(fc3_BN)
    mlp_BN_acc = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(fc3_BN, 1), tf.argmax(y, 1)), tf.float32))
    mlp_BN_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(fc3_BN, y))
    mlp_BN_optimizer = tf.train.AdamOptimizer(
        FLAGS.learning_rate).minimize(mlp_BN_loss)

    # Training
    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())


        accs = []
        BN_accs = []
        losses = []
        BN_losses = []

        for batch_num in range(10000):
            batch = data.train.next_batch(64)

            output_feed = [mlp_acc, mlp_BN_acc,
                           mlp_loss, mlp_BN_loss,
                          mlp_optimizer, mlp_BN_optimizer]
            input_feed = {X: batch[0], y: batch[1]}
            acc, acc_BN, loss, loss_BN, _, _ = sess.run(
                output_feed, input_feed)

            losses.append(loss)
            BN_losses.append(loss_BN)
            accs.append(acc)
            BN_accs.append(acc_BN)

        print "MLP Accuracy: %.3f" % np.mean(accs)
        print "MLP w/ BN Accuracy: %.3f" % np.mean(BN_accs)

        fig, ax = plt.subplots()
        ax.plot(range(len(losses)), losses, label='Without BN')
        ax.plot(range(len(BN_losses)), BN_losses, label='With BN')
        ax.set_xlabel('Training steps')
        ax.set_ylabel('Loss')
        ax.set_title('Loss w/wout BN')
        ax.legend(loc=1)
        plt.show()


        # Inference (1 sample at a time = usual inference)
        accs = []
        for sample_num in range(20):
            batch = data.train.next_batch(1)
            output_feed = [mlp_BN_acc]
            input_feed = {X: batch[0], y: batch[1]}
            acc_BN = sess.run(output_feed, input_feed)
            accs.append(acc_BN)
        print "Inference accuracy: %.3f" % np.mean(accs)








if __name__ == '__main__':
    FLAGS = parameters()
    train(FLAGS)

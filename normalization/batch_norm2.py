import tensorflow as tf
import seaborn as sns
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import (
    input_data,
)

from tensorflow.contrib.layers import (
    batch_norm
)

data = input_data.read_data_sets('MNIST_data', one_hot=True)

# Intialize weights with numpy so we can pass same to both MLP and MLP_BN
W1_init = np.random.normal(size=(784,100)).astype(np.float32)
W2_init = np.random.normal(size=(100,100)).astype(np.float32)
W3_init = np.random.normal(size=(100,10)).astype(np.float32)

class parameters():

    def __init__(self):
        self.num_epochs = 10
        self.batch_size = 64

        self.epsilon = 1e-3
        self.learning_rate = 0.01

        self.record_every = 50
        self.ckpt_dir = 'checkpoints'

class model(object):

    def __init__(self, FLAGS):

        self.X = tf.placeholder(tf.float32, [None, None], 'X')
        self.y = tf.placeholder(tf.float32, [None, None], 'y')
        self.is_training_ph = tf.placeholder(tf.bool, name='is_training')

        # MLP
        self.W1 = tf.Variable(W1_init)
        self.b1 = tf.Variable(tf.zeros([100]))
        self.z1 = tf.matmul(self.X, self.W1)+ self.b1
        self.fc1 = tf.nn.relu(self.z1)

        self.W2 = tf.Variable(W2_init)
        self.b2 = tf.Variable(tf.zeros([100]))
        self.z2 = tf.matmul(self.fc1, self.W2)+ self.b2
        self.fc2 = tf.nn.relu(self.z2)

        self.W3 = tf.Variable(W3_init)
        self.b3 = tf.Variable(tf.zeros([10]))
        self.fc3  = tf.matmul(self.fc2,self.W3)+ self.b3

        self.mlp_probabilities = tf.nn.softmax(self.fc3)
        self.mlp_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.fc3, 1),
                tf.argmax(self.y, 1)), tf.float32))
        self.mlp_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.fc3, self.y))
        self.mlp_optimizer = tf.train.AdamOptimizer(
            FLAGS.learning_rate).minimize(self.mlp_loss)

        # MLP with Batchnorm
        # Naive BN layer
        self.scale1 = tf.Variable(tf.ones([100]))
        self.shift1 = tf.Variable(tf.zeros([100]))
        self.W1_BN = tf.Variable(W1_init)
        self.b1_BN = tf.Variable(tf.zeros([100]))
        self.z1_BN = tf.matmul(self.X, self.W1_BN)+ self.b1_BN

        with tf.variable_scope('BN_1') as BN_1:
            self.BN1 = tf.cond(self.is_training_ph,
                lambda: batch_norm(
                    self.z1_BN, is_training=True, center=True,
                    scale=True, activation_fn=tf.nn.relu,
                    updates_collections=None, scope=BN_1),
                lambda: batch_norm(
                    self.z1_BN, is_training=False, center=True,
                    scale=True, activation_fn=tf.nn.relu,
                    updates_collections=None, scope=BN_1, reuse=True))

        self.fc1_BN = tf.nn.relu(self.BN1)

        # TF BN layer
        self.scale2 = tf.Variable(tf.ones([100]))
        self.shift2 = tf.Variable(tf.zeros([100]))
        self.W2_BN = tf.Variable(W2_init)
        self.b2_BN = tf.Variable(tf.zeros([100]))
        self.z2_BN = tf.matmul(self.fc1_BN, self.W2_BN)+ self.b2_BN

        with tf.variable_scope('BN_2') as BN_2:
            self.BN2 = tf.cond(self.is_training_ph,
                lambda: batch_norm(
                    self.z2_BN, is_training=True, center=True,
                    scale=True, activation_fn=tf.nn.relu,
                    updates_collections=None, scope=BN_2),
                lambda: batch_norm(
                    self.z2_BN, is_training=False, center=True,
                    scale=True, activation_fn=tf.nn.relu,
                    updates_collections=None, scope=BN_2, reuse=True))

        self.fc2_BN = tf.nn.relu(self.BN2)

        self.W3_BN = tf.Variable(W3_init)
        self.b3_BN = tf.Variable(tf.zeros([10]))
        self.fc3_BN = tf.matmul(self.fc2_BN, self.W3_BN)+ self.b3_BN

        self.mlp_BN_probabilities = tf.nn.softmax(self.fc3_BN)
        self.mlp_BN_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.fc3_BN, 1),
                tf.argmax(self.y, 1)), tf.float32))
        self.mlp_BN_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.fc3_BN, self.y))
        self.mlp_BN_optimizer = tf.train.AdamOptimizer(
            FLAGS.learning_rate).minimize(self.mlp_BN_loss)

        # Components for model saving
        self.global_step = tf.Variable(0, trainable=False)
        self.saver = tf.train.Saver(tf.all_variables())

def create_model(sess, FLAGS):

    tf_model = model(FLAGS)

    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Restoring old model parameters from %s" %
                             ckpt.model_checkpoint_path)
        tf_model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created new model.")
        sess.run(tf.initialize_all_variables())

    return tf_model


def train(FLAGS):
    """
    Train our MLP and MLP_BN.
    """

    # Training
    with tf.Session() as sess:

        model = create_model(sess, FLAGS)

        accs = []
        BN_accs = []
        losses = []
        BN_losses = []

        for batch_num in range(10000):
            batch = data.train.next_batch(64)

            output_feed = [model.mlp_acc, model.mlp_BN_acc,
                           model.mlp_loss, model.mlp_BN_loss,
                          model.mlp_optimizer, model.mlp_BN_optimizer]
            input_feed = {model.X: batch[0], model.y: batch[1],
                          model.is_training_ph: True}
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
            output_feed = [model.mlp_BN_acc]
            input_feed = {model.X: batch[0], model.y: batch[1],
                          model.is_training_ph: False}
            acc_BN = sess.run(output_feed, input_feed)
            accs.append(acc_BN)

        print "Inference accuracy: %.3f" % np.mean(accs)

        # Save the trained model
        if not os.path.isdir(FLAGS.ckpt_dir):
            os.makedirs(FLAGS.ckpt_dir)
        checkpoint_path = os.path.join(FLAGS.ckpt_dir, "model.ckpt")
        print "Saving the model."
        model.saver.save(sess, checkpoint_path,
                         global_step=model.global_step)


if __name__ == '__main__':
    FLAGS = parameters()
    train(FLAGS)

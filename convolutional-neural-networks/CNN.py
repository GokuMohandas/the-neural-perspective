import tensorflow as tf
import numpy as np
import input_data
import os

class parameters():

    def __init__(self):
        self.batch_size = 128
        self.num_epochs = 10
        self.ckpt_dir = "CNN_ckpt_dir" # save models here

def load_data():
    # Load the data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trainX, trainY, testX, testY = mnist.train.images, mnist.train.labels, \
                             mnist.test.images, mnist.test.labels
    trainX = trainX.reshape(-1, 28, 28, 1)
    testX = testX.reshape(-1, 28, 28, 1)
    return trainX, trainY, testX, testY

def generate_batches(batch_size, X, y):

    # Create batches
    num_batches = len(X) // batch_size
    data_X = np.zeros([num_batches, batch_size, 28, 28, 1], dtype=np.float32)
    data_y = np.zeros([num_batches, batch_size, 10], dtype=np.float32)
    for batch_num in range(num_batches):
        data_X[batch_num] = X[batch_num*batch_size:(batch_num+1)*batch_size]
        data_y[batch_num] = y[batch_num*batch_size:(batch_num+1)*batch_size]
        yield data_X[batch_num], data_y[batch_num]

def generate_epochs(num_epochs, batch_size, X, y):

    for epoch_num in range(num_epochs):
        yield generate_batches(batch_size, X, y)

def cnn_operations(X, w, w2, w3, w4, w_o, 
                dropout_value_conv, dropout_value_hidden):

    l1a = tf.nn.relu(tf.nn.conv2d(X, w, 
        strides=[1,1,1,1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1,2,2,1], 
        strides=[1,2,2,1], padding='SAME')
    l1 = tf.nn.dropout(l1, dropout_value_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, 
        strides=[1,1,1,1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1,2,2,1], 
        strides=[1,2,2,1], padding='SAME')
    l2 = tf.nn.dropout(l2, dropout_value_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, 
        strides=[1,1,1,1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1], 
        strides=[1,2,2,1], padding='SAME')
    l3 = tf.reshape(l3, 
        [-1, w4.get_shape().as_list()[0]]) # flatten to shape(?, 2048)
    l3 = tf.nn.dropout(l3, dropout_value_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, dropout_value_hidden)

    return tf.matmul(l4, w_o)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

class cnn_model(object):

    def __init__(self):

        # Placeholders
        self.X = tf.placeholder("float", [None, 28, 28, 1])
        self.y = tf.placeholder("float", [None, 10])
        self.dropout_value_conv = tf.placeholder("float")
        self.dropout_value_hidden = tf.placeholder("float")

        # Initalize weights
        w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
        w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
        w3 = init_weights([3, 3, 64, 128])    # 3x3x64 conv, 128 outputs
        w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 = 2048 inputs, 625 outputs
        w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

        self.logits = cnn_operations(self.X, w, w2, w3, w4, w_o, 
                    self.dropout_value_conv, self.dropout_value_hidden)
        self.cost = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                                    self.logits, self.y))
        self.optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(
                                                    self.cost)

        # Accuracy
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), 
                                    tf.argmax(self.logits, 1))
        self.accuracy = tf.reduce_mean(tf.cast(
                            self.correct_prediction, tf.float32))

        # Components for model saving
        self.global_step = tf.Variable(0, trainable=False)
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, sess, batch_X, batch_y, 
        dropout_value_conv, dropout_value_hidden, 
        forward_only=True):

        input_feed = {self.X: batch_X, self.y: batch_y,
                self.dropout_value_conv: dropout_value_conv,
                self.dropout_value_hidden: dropout_value_hidden}

        if not forward_only:
            output_feed = [self.logits, self.cost, 
                           self.accuracy, self.optimizer]
        else:
            output_feed = [self.cost, self.accuracy]

        outputs = sess.run(output_feed, input_feed)

        if not forward_only:
            return outputs[0], outputs[1], outputs[2], outputs[3]
        else:
            return outputs[0], outputs[1]

def create_model(sess, FLAGS):

    model = cnn_model()

    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Restoring old model parameters from %s" % 
                             ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created new model.")
        sess.run(tf.initialize_all_variables())

    return model

def train(FLAGS):

    with tf.Session() as sess:

        model = create_model(sess, FLAGS)
        trainX, trainY, testX, testY = load_data()

        # Training
        for epoch_num, epoch in enumerate(
                                generate_epochs(FLAGS.num_epochs, 
                                                FLAGS.batch_size,
                                                trainX,
                                                trainY)):
            train_cost = []
            train_accuracy = []
            print "Training in progress..."
            for batch_num, (input_X, labels_y) in enumerate(epoch):
                logits, cost, accuracy, _ = model.step(sess, 
                                                       input_X, labels_y,
                                                       dropout_value_conv=0.8,
                                                       dropout_value_hidden=0.5,
                                                       forward_only=False)
                train_cost.append(cost)
                train_accuracy.append(accuracy)

            print "Training:"
            print "Epoch: %i, batch: %i, cost: %.3f, accuarcy: %.3f" % (
                    epoch_num, batch_num, 
                    np.mean(train_cost), np.mean(train_accuracy))

            # Validation
            for epoch_num, epoch in enumerate(generate_epochs(
                                                num_epochs=1, 
                                                batch_size=FLAGS.batch_size, 
                                                X=testX, 
                                                y=testY)):
                test_cost = []
                test_accuracy = []
                for batch_num, (input_X, labels_y) in enumerate(epoch):
                    cost, accuracy = model.step(sess, 
                                                input_X, labels_y,
                                                dropout_value_conv=0.0,
                                                dropout_value_hidden=0.0,
                                                forward_only=True)
                    test_cost.append(cost)
                    test_accuracy.append(accuracy)

                print "Validation:"
                print "Epoch: %i, batch: %i, cost: %.3f, accuarcy: %.3f" % (
                    epoch_num, batch_num, 
                    np.mean(test_cost), np.mean(test_accuracy))

            # Save checkpoint every epoch.
            if not os.path.isdir(FLAGS.ckpt_dir):
                os.makedirs(FLAGS.ckpt_dir)
            checkpoint_path = os.path.join(FLAGS.ckpt_dir, "model.ckpt")
            print "Saving the model."
            model.saver.save(sess, checkpoint_path, 
                             global_step=model.global_step)

if __name__ == '__main__':
    FLAGS = parameters()
    train(FLAGS)
















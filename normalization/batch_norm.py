import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import input_data

class parameters():
    """
    Hyperparameteres for the net.
    """
    def __init__(self):
        self.num_epochs = 10
        self.batch_size = 64

        self.max_gradient_norm = 5.0
        self.learning_rate = 0.01

def load_data():
    """
    Load the MNIST data.
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trainX, trainY, testX, testY = mnist.train.images, mnist.train.labels, \
        mnist.test.images, mnist.test.labels

    return trainX, trainY, testX, testY

def generate_epoch(X, y, num_epochs, batch_size):

    for epoch_num in range(num_epochs):
        yield generate_batch(X, y, batch_size)

def generate_batch(X, y, batch_size):

    data_size = len(X)

    num_batches = (data_size // batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield X[start_index:end_index], y[start_index:end_index]

def mlp(inputs, W1, b1, W2, b2, W3, b3):
    """
    Normal MLP (no BN).
    """
    with tf.variable_scope('mlp') as scope:
        scope.reuse_variables()
        fc1 = tf.nn.relu(tf.matmul(inputs, W1) + b1)
        fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)
        fc3 = tf.nn.relu(tf.matmul(fc2, W3) + b3)
        return fc3, fc2

def mlp_BN(inputs, scale1, shift1, scale2, shift2,
           pop_mean1, pop_var1, pop_mean2, pop_var2,
           W1, b1, W2, b2, W3, b3):
    """
    MLP with BN
    """

    # Stability in BN
    epsilon = 1e-3

    # For exponential moving average (ema)
    decay = 0.999

    with tf.variable_scope('mlp_norm') as scope:
        scope.reuse_variables()
        fc1 = tf.matmul(inputs, W1) + b1

        # BN to 1st layer using naive implementation
        mean1, var1 = tf.nn.moments(fc1, [0])
        fc1 = (fc1 - mean1) / tf.sqrt(var1 + epsilon)
        fc1 = scale1 * fc1 + shift1
        fc1 = tf.nn.relu(fc1)

        fc2 = tf.matmul(fc1, W2) + b2

        # BN to 2nd layer using TF
        mean2, var2 = tf.nn.moments(fc2, [0])
        fc2 = tf.nn.batch_normalization(
            fc2, mean2, var2, shift2, scale2, epsilon)
        fc2 = tf.nn.relu(fc2)

        fc3 = tf.nn.relu(tf.matmul(fc2, W3) + b3)

        # Update the population mean and var
        pop_mean1 = pop_mean1 * decay + mean1 * (1-decay)
        pop_var1 = pop_var1 * decay + \
            (FLAGS.batch_size / (1-FLAGS.batch_size)) * (var1 * (1-decay))
        pop_mean2 = pop_mean2 * decay + mean1 * (1-decay)
        pop_var2 = pop_var2 * decay + \
            (FLAGS.batch_size / (1-FLAGS.batch_size)) * (var2 * (1-decay))

        return fc3, fc1


class model():

    def __init__(self, FLAGS):

        self.X = tf.placeholder(tf.float32, shape=[None, None], name='X')
        self.y = tf.placeholder(tf.float32, shape=[None, None], name='y')

        with tf.variable_scope('mlp'):
            W1 = tf.get_variable('W1', shape=[784, 100])
            b1 = tf.get_variable('b1', shape=[100])
            W2 = tf.get_variable('W2', shape=[100, 100])
            b2 = tf.get_variable('b2', shape=[100])
            W3 = tf.get_variable('W3', shape=[100, 10])
            b3 = tf.get_variable('b3', shape=[10])
            self.mlp_weights = [W1, b1, W2, b2, W3, b3]
            self.mlp_logits, self.fc2 = mlp(self.X, W1, b1, W2, b2, W3, b3)

        self.mlp_softmax = tf.nn.softmax(self.mlp_logits)
        self.mlp_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            self.mlp_logits, self.y))

        mlp_trainable_vars =  tf.get_collection(tf.GraphKeys.VARIABLES,
            scope='mlp')
        # clip the gradient to avoid vanishing or blowing up gradients
        mlp_grads, self.mlp_norm = tf.clip_by_global_norm(
            tf.gradients(self.mlp_cost, mlp_trainable_vars),
            FLAGS.max_gradient_norm)
        self.mlp_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate
            ).apply_gradients(zip(mlp_grads, mlp_trainable_vars))

        self.mlp_optimizer = tf.train.GradientDescentOptimizer(
            FLAGS.learning_rate).minimize(self.mlp_cost)

        with tf.variable_scope('mlp_BN'):

            # Scale and shift parameters
            scale1 = tf.get_variable('alpha1', shape=[100])
            shift1 = tf.get_variable('beta1', shape=[100])
            scale2 = tf.get_variable('alpha2', shape=[100])
            shift2 = tf.get_variable('beta2', shape=[100])

            # Population mean and var
            pop_mean1 = tf.get_variable('pop_mean1',
                shape=[100], initializer=tf.constant_initializer(0.0),
                trainable=False)
            pop_var1 = tf.get_variable('pop_var1',
                shape=[100], initializer=tf.constant_initializer(0.0),
                trainable=False)
            pop_mean2 = tf.get_variable('pop_mean2',
                shape=[100], initializer=tf.constant_initializer(0.0),
                trainable=False)
            pop_var2 = tf.get_variable('pop_var2',
                shape=[100], initializer=tf.constant_initializer(0.0),
                trainable=False)

            W1 = tf.get_variable('W1', shape=[784, 100])
            b1 = tf.get_variable('b1', shape=[100])
            W2 = tf.get_variable('W2', shape=[100, 100])
            b2 = tf.get_variable('b2', shape=[100])
            W3 = tf.get_variable('W3', shape=[100, 10])
            b3 = tf.get_variable('b3', shape=[10])
            self.mlp_BN_weights = [W1, b1, W2, b2, W3, b3]
            self.mlp_BN_logits, self.fc2_BN = mlp_BN(
                self.X, scale1, shift1, scale2, shift2,
                pop_mean1, pop_var1, pop_mean2, pop_var2,
                W1, b1, W2, b2, W3, b3)

        self.mlp_BN_softmax = tf.nn.softmax(self.mlp_BN_logits)
        self.mlp_BN_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.mlp_BN_logits, self.y))

        mlp_BN_trainable_vars =  tf.get_collection(tf.GraphKeys.VARIABLES,
            scope='mlp_BN')

        # clip the gradient to avoid vanishing or blowing up gradients
        mlp_BN_grads, self.mlp_BN_norm = tf.clip_by_global_norm(
            tf.gradients(self.mlp_BN_cost, mlp_BN_trainable_vars),
            FLAGS.max_gradient_norm)
        self.mlp_BN_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate
            ).apply_gradients(zip(mlp_BN_grads, mlp_BN_trainable_vars))

        self.mlp_BN_optimizer = tf.train.GradientDescentOptimizer(
            FLAGS.learning_rate).minimize(self.mlp_BN_cost)

        self.accuracy_BN = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.mlp_BN_logits, 1),
                                    tf.argmax(self.y, 1)), tf.float32))


    def step(self, sess, batch_X, batch_y, forward_only=True):

        input_feed = {self.X: batch_X, self.y:batch_y}
        if not forward_only:
            output_feed = [self.mlp_cost, self.mlp_norm, self.fc2,
                           self.mlp_optimizer,
                           self.mlp_BN_cost, self.mlp_BN_norm, self.fc2_BN,
                           self.mlp_BN_optimizer,
                           self.accuracy_BN]
            outputs = sess.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2], outputs[3], \
                    outputs[4], outputs[5], outputs[6], outputs[7], \
                    outputs[8]
        # Inference
        else:
            output_feed = [self.accuracy_BN]
            outputs = sess.run(output_feed, input_feed)
            return outputs[0]

def create_model(sess, FLAGS):

    tf_model = model(FLAGS)
    print("Created new model.")
    sess.run(tf.initialize_all_variables())

    return tf_model

def train(FLAGS):
    """
    Function to train our net.
    """

    # Load the data
    X_train, y_train, X_test, y_test = load_data()

    # Need to use same initial weights for unbiased comparsion.
    # SCALE is very important when initializing weights
    W1_init = np.random.normal(scale=0.1, size=(784, 100)).astype(np.float32)
    b1_init = np.zeros([100]).astype(np.float32)
    W2_init = np.random.normal(scale=0.1, size=(100, 100)).astype(np.float32)
    b2_init = np.zeros([100]).astype(np.float32)
    W3_init   = np.random.normal(scale=0.1, size=(100, 10)).astype(np.float32)
    b3_init = np.zeros([10]).astype(np.float32)
    W_inits = [W1_init, b1_init, W2_init, b2_init, W3_init, b3_init]

    with tf.Session() as sess:

        # Model
        model = create_model(sess, FLAGS)

        # Initialize a variables
        for i, W in enumerate(model.mlp_weights):
            sess.run(W.assign(W_inits[i]))
        for i, W in enumerate(model.mlp_BN_weights):
            sess.run(W.assign(W_inits[i]))

        # Store variables for plots
        costs = []
        norms = []
        cost_BNs = []
        norm_BNs = []
        fc2s = []
        fc2_BNs = []

        # Train the model
        for epoch_num, epoch in enumerate(
            generate_epoch(X_train, y_train,
                FLAGS.num_epochs, FLAGS.batch_size)):

            train_acc = []
            test_acc = []

            for batch_num, (batch_X, batch_y)  in enumerate(epoch):

                # Train
                cost, norm, fc2, _, cost_BN, norm_BN, fc2_BN, _, \
                BN_train_accuarcy = \
                    model.step(sess, batch_X, batch_y, forward_only=False)


                test_batch_X = X_test[:FLAGS.batch_size*10]
                test_batch_y = y_test[:FLAGS.batch_size*10]
                # Inference
                BN_test_accuracy = model.step(sess, test_batch_X, test_batch_y,
                    forward_only=True)

                train_acc.append(BN_train_accuarcy)
                test_acc.append(BN_test_accuracy)

            print "Train accuracy: ", np.mean(train_acc)
            print "Test accuracy: ", np.mean(test_acc)

            costs.append(cost)
            norms.append(norm)
            cost_BNs.append(cost_BN)
            norm_BNs.append(norm_BN)
            fc2s.append(np.mean(fc2))
            fc2_BNs.append(np.mean(fc2_BN))

            print "Epoch %i complete" % epoch_num

        # Plot the results
        # Cost
        fig, ax = plt.subplots()
        ax.plot(range(0, len(costs)), costs, label='Without BN')
        ax.plot(range(0, len(cost_BNs)), cost_BNs, label='With BN')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Cost')
        ax.set_title('Batch Normalization Cost')
        ax.legend(loc=4)
        plt.show()

        # Gradient norm
        fig, ax = plt.subplots()
        ax.plot(range(0, len(norms)), norms, label='Without BN')
        ax.plot(range(0, len(norm_BNs)), norm_BNs, label='With BN')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Batch Normalization Gradient Norm')
        ax.legend(loc=4)
        plt.show()

        # Inputs to first neuron in third layer
        fig, ax = plt.subplots()
        ax.plot(range(0, len(fc2s)), fc2s, label='Without BN')
        ax.plot(range(0, len(fc2_BNs)), fc2_BNs, label='With BN')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Neuron inputs')
        ax.set_title('Batch Normalization Neuron Inputs')
        ax.legend(loc=4)
        plt.show()




if __name__ == '__main__':
    FLAGS = parameters()
    train(FLAGS)


import numpy as np
import tensorflow as tf
import sys
import os
import random
import cPickle

class parameters():

    def __init__(self):

        self.DATA_FILE = 'data/shakespeare.txt'
        self.CKPT_DIR = 'char_RNN_ckpt_dir'
        self.encoding = 'utf-8'
        self.SAMPLE_EVERY = 10 # save model every epoch
        self.TRAIN_RATIO = 0.8
        self.VALID_RATIO = 0.1

        self.NUM_EPOCHS = 100
        self.BATCH_SIZE = 1000
        self.SEQ_LEN = 300
        self.MODEL = 'rnn'
        self.NUM_HIDDEN_UNITS = 128
        self.NUM_LAYERS = 1
        self.DROPOUT = 0.5

        self.GRAD_CLIP = 5.0
        self.LEARNING_RATE = 0.002
        self.DECAY_RATE = 0.97

        self.SAMPLE_LEN = 500
        self.SEED_TOKENS = "Thou "
        self.SAMPLE_EVERY = 10 # general sample every epoch
        self.SAMPLE_TYPE = 1 # 0=argmax, 1=temperature based
        self.TEMPERATURE = 0.04

def generate_data(FLAGS):

    data = open(FLAGS.DATA_FILE, "r").read()
    chars = list(set(data))
    char_to_idx = {char:i for i, char in enumerate(chars)}
    idx_to_char = {i:char for i, char in enumerate(chars)}

    FLAGS.DATA_SIZE = len(data)
    FLAGS.NUM_CLASSES = len(chars)
    FLAGS.char_to_idx = char_to_idx
    FLAGS.idx_to_char = idx_to_char
    print "\nTotal %i characters with %i unique tokens." % \
        (FLAGS.DATA_SIZE, FLAGS.NUM_CLASSES)

    X = [FLAGS.char_to_idx[char] for char in data]
    y = X[1:]
    y[-1] = X[0]

    # Split into train, valid and test sets
    train_last_index = int(FLAGS.DATA_SIZE*FLAGS.TRAIN_RATIO)
    valid_first_index = train_last_index + 1
    valid_last_index = valid_first_index + \
        int(FLAGS.DATA_SIZE*FLAGS.VALID_RATIO)
    test_first_index = valid_last_index + 1

    FLAGS.train_X = X[:train_last_index]
    FLAGS.train_y = y[:train_last_index]
    FLAGS.valid_X = X[valid_first_index:valid_last_index]
    FLAGS.valid_y = y[valid_first_index:valid_last_index]
    FLAGS.test_X = X[test_first_index:]
    FLAGS.test_y = y[test_first_index:]

    FLAGS.NUM_BATCHES = len(FLAGS.train_X) // FLAGS.BATCH_SIZE

    return FLAGS

def generate_batch(FLAGS, raw_X, raw_y):

    # Create batches from raw data
    num_batches = len(raw_X) // FLAGS.BATCH_SIZE # tokens per batch
    data_X = np.zeros([num_batches, FLAGS.BATCH_SIZE], dtype=np.int32)
    data_y = np.zeros([num_batches, FLAGS.BATCH_SIZE], dtype=np.int32)
    for i in range(num_batches):
        data_X[i, :] = raw_X[FLAGS.BATCH_SIZE * i: FLAGS.BATCH_SIZE * (i+1)]
        data_y[i, :] = raw_y[FLAGS.BATCH_SIZE * i: FLAGS.BATCH_SIZE * (i+1)]

    # Even though we have tokens per batch,
    # We only want to feed in <SEQ_LEN> tokens at a time
    feed_size = FLAGS.BATCH_SIZE // FLAGS.SEQ_LEN
    for i in range(feed_size):
        X = data_X[:, i * FLAGS.SEQ_LEN:(i+1) * FLAGS.SEQ_LEN]
        y = data_y[:, i * FLAGS.SEQ_LEN:(i+1) * FLAGS.SEQ_LEN]
        yield (X, y)

def generate_epochs(FLAGS, raw_X, raw_y):

    for i in range(FLAGS.NUM_EPOCHS):
        yield generate_batch(FLAGS, raw_X, raw_y)

def rnn_cell(FLAGS):

    # Get the cell type
    if FLAGS.MODEL == 'rnn':
        rnn_cell_type = tf.nn.rnn_cell.BasicRNNCell
    elif FLAGS.MODEL == 'gru':
        rnn_cell_type = tf.nn.rnn_cell.GRUCell
    elif FLAGS.MODEL == 'lstm':
        rnn_cell_type = tf.nn.rnn_cell.BasicLSTMCell
    else:
        raise Exception("Choose a valid RNN unit type.")

    # Single cell
    single_cell = rnn_cell_type(FLAGS.NUM_HIDDEN_UNITS)

    # Dropout
    single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell,
        output_keep_prob=1-FLAGS.DROPOUT)

    # Each state as one cell
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * FLAGS.NUM_LAYERS)

    return stacked_cell

def rnn_inputs(FLAGS, input_data):
    with tf.variable_scope('rnn_inputs', reuse=True):
        W_input = tf.get_variable("W_input",
            [FLAGS.NUM_CLASSES, FLAGS.NUM_HIDDEN_UNITS])

    # <BATCH_SIZE, seq_len, num_hidden_units>
    embeddings = tf.nn.embedding_lookup(W_input, input_data)
    # <seq_len, BATCH_SIZE, num_hidden_units>
    # BATCH_SIZE will be in columns bc we feed in row by row into RNN.
    # 1st row = 1st tokens from each batch
    #inputs = [tf.squeeze(i, [1]) for i in tf.split(1, FLAGS.SEQ_LEN, embeddings)]
    # NO NEED if using dynamic_rnn(time_major=False)
    return embeddings

def rnn_softmax(FLAGS, outputs):
    with tf.variable_scope('rnn_softmax', reuse=True):
        W_softmax = tf.get_variable("W_softmax",
            [FLAGS.NUM_HIDDEN_UNITS, FLAGS.NUM_CLASSES])
        b_softmax = tf.get_variable("b_softmax", [FLAGS.NUM_CLASSES])

    logits = tf.matmul(outputs, W_softmax) + b_softmax
    return logits

class model(object):

    def __init__(self, FLAGS):

        ''' Data placeholders '''
        self.input_data = tf.placeholder(tf.int32, [None, None])
        self.targets = tf.placeholder(tf.int32, [None, None])

        ''' RNN cell '''
        self.stacked_cell = rnn_cell(FLAGS)
        self.initial_state = self.stacked_cell.zero_state(
            FLAGS.NUM_BATCHES, tf.float32)

        ''' Inputs to RNN '''
        # Embedding (aka W_input weights)
        with tf.variable_scope('rnn_inputs'):
            W_input = tf.get_variable("W_input",
                [FLAGS.NUM_CLASSES, FLAGS.NUM_HIDDEN_UNITS])
        inputs = rnn_inputs(FLAGS, self.input_data)

        ''' Outputs from RNN '''
        # Outputs: <seq_len, BATCH_SIZE, num_hidden_units>
        # state: <BATCH_SIZE, num_layers*num_hidden_units>
        outputs, state = tf.nn.dynamic_rnn(cell=self.stacked_cell, inputs=inputs,
                                           initial_state=self.initial_state)

        # <seq_len*BATCH_SIZE, num_hidden_units>
        outputs = tf.reshape(tf.concat(1, outputs), [-1, FLAGS.NUM_HIDDEN_UNITS])

        ''' Process RNN outputs '''
        with tf.variable_scope('rnn_softmax'):
            W_softmax = tf.get_variable("W_softmax",
                [FLAGS.NUM_HIDDEN_UNITS, FLAGS.NUM_CLASSES])
            b_softmax = tf.get_variable("b_softmax", [FLAGS.NUM_CLASSES])
        # Logits
        self.logits = rnn_softmax(FLAGS, outputs)
        self.probabilities = tf.nn.softmax(self.logits)

        ''' Loss '''
        y_as_list = tf.reshape(self.targets, [-1])
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.logits, y_as_list))
        self.final_state = state

        ''' Optimization '''
        self.lr = tf.Variable(0.0, trainable=False)
        trainable_vars = tf.trainable_variables()
        # glip the gradient to avoid vanishing or blowing up gradients
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars),
                                          FLAGS.GRAD_CLIP)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_optimizer = optimizer.apply_gradients(
            zip(self.grads, trainable_vars))

        # Components for model saving
        self.global_step = tf.Variable(0, trainable=False)
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, sess, batch_X, batch_y, initial_state=None):

        if initial_state == None:
            input_feed = {self.input_data: batch_X,
                          self.targets: batch_y}
        else:
            input_feed = {self.input_data: batch_X,
                          self.targets: batch_y,
                          self.initial_state: initial_state}

        output_feed = [self.loss,
                       self.final_state,
                       self.logits,
                       self.train_optimizer,
                       self.grads]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]

    def sample(self, sess, FLAGS, sampling_type=1):

        # Process state for given SEED_TOKENS
        for char_num, char in enumerate(FLAGS.SEED_TOKENS[:-1]):
            word = np.array(FLAGS.char_to_idx[char]).reshape(1,1)
            if char_num == 0:
                state = np.zeros((FLAGS.NUM_BATCHES,
                    FLAGS.NUM_LAYERS * FLAGS.NUM_HIDDEN_UNITS))
            state = sess.run(self.final_state,
                feed_dict={self.input_data:word, self.initial_state:state})

            state = np.array(state).reshape((1, np.shape(state)[-1]))

        # Sample text for <sample_len> characters
        sample = FLAGS.SEED_TOKENS
        prev_char = sample[-1]
        for word_num in range(0, FLAGS.SAMPLE_LEN):
            word = np.array(FLAGS.char_to_idx[prev_char]).reshape(1,1)
            probs, state = sess.run([self.probabilities,
                                     self.final_state],
                            feed_dict={self.input_data:word,
                                       self.initial_state:state})
            state = np.array(state).reshape((1, np.shape(state)[-1]))

            # probs[0] bc probs is 2D array with just one item
            next_char_dist = probs[0]

            # scale the distribution
            # for creativity, higher temperatures produce
            # more nonexistent words BUT more creative samples
            next_char_dist /= FLAGS.TEMPERATURE
            next_char_dist = np.exp(next_char_dist)
            next_char_dist /= sum(next_char_dist)

            if FLAGS.SAMPLE_TYPE == 0:
                choice_index = np.argmax(next_char_dist)
            elif FLAGS.SAMPLE_TYPE  == 1:
                choice_index = -1
                point = random.random()
                weight = 0.0
                for index in range(0, FLAGS.NUM_CLASSES):
                    weight += next_char_dist[index]
                    if weight >= point:
                        choice_index = index
                        break
            else:
                raise ValueError("Pick a valid sampling_type!")

            sample += FLAGS.idx_to_char[choice_index]
            prev_char = sample[-1]

        return sample


def create_model(sess, FLAGS):

    char_model = model(FLAGS)

    ckpt = tf.train.get_checkpoint_state(FLAGS.CKPT_DIR)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Restoring old model parameters from %s" %
                        ckpt.model_checkpoint_path)
        char_model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created new model.")
        sess.run(tf.initialize_all_variables())

    return char_model

def train(FLAGS):

    with tf.Session() as sess:

        model = create_model(sess, FLAGS)
        state = None

        for epoch_num, epoch in enumerate(generate_epochs(FLAGS,
                    FLAGS.train_X, FLAGS.train_y)):
            train_loss = []

            # Assign/update learning rate
            sess.run(tf.assign(model.lr, FLAGS.LEARNING_RATE *
                (FLAGS.DECAY_RATE ** epoch_num)))

            # Training
            for minibatch_num, (X, y) in enumerate(epoch):
                loss, state, logits, _, grads = model.step(sess, X, y, state)
                train_loss.append(loss)

            print np.shape(grads)
            print np.shape(grads[0])
            print grads[0]
            print see_gradients_above
            print "Training loss %.3f" % np.mean(train_loss)

            # Save checkpoint every epoch.
            if not os.path.isdir(FLAGS.CKPT_DIR):
                os.makedirs(FLAGS.CKPT_DIR)
            checkpoint_path = os.path.join(FLAGS.CKPT_DIR, "model.ckpt")
            print "Saving the model at Epoch %i." % epoch_num
            model.saver.save(sess, checkpoint_path,
                global_step=model.global_step)

            if epoch_num%FLAGS.SAMPLE_EVERY == 0:
                os.system('python tf_char_rnn.py sample')

def sample(FLAGS):

    with tf.Session() as sess:

        model = create_model(sess, FLAGS)
        print model.sample(sess, FLAGS, sampling_type=1)


if __name__ == '__main__':

    if sys.argv[1] == 'train':
        FLAGS = parameters()
        FLAGS = generate_data(FLAGS)
        train(FLAGS)
    elif sys.argv[1] == 'sample':
        FLAGS = parameters()
        FLAGS = generate_data(FLAGS)
        FLAGS.NUM_BATCHES = 1
        sample(FLAGS)
    else:
        print "Please choose train or sample"

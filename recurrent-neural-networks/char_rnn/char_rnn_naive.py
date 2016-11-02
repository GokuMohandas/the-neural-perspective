import numpy as np
import tensorflow as tf
import random

class parameters():

    def __init__(self):
        self.NUM_EPOCHS = 2001
        self.SEQ_LEN = 200 # of tokens per feed for each minibatch row
        self.BATCH_SIZE = 1000
        self.NUM_HIDDEN_UNITS = 100 # num hidden units per state
        self.LEARNING_RATE = 1e-4
        self.FILE = "data/shakespeare.txt"
        self.ENCODING = 'utf-8'

        self.START_TOKEN = 'Thou'
        self.PREDICTION_LENGTH = 50
        self.TEMPERATURE = 0.04

def generate_data(FLAGS):

    X = [FLAGS.char_to_idx[char] for char in FLAGS.data]
    y = X[1:]
    y[-1] = X[0]

    return X, y

def generate_batch(FLAGS, raw_data):
    raw_X, raw_y = raw_data
    data_length = len(raw_X)

    # Create batches from raw data
    num_batches = FLAGS.DATA_SIZE // FLAGS.BATCH_SIZE # tokens per batch
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

def generate_epochs(FLAGS):
    for i in range(FLAGS.NUM_EPOCHS):
        yield generate_batch(FLAGS, generate_data(FLAGS))

def rnn_cell(FLAGS, rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W_input = tf.get_variable('W_input', 
            [FLAGS.NUM_CLASSES, FLAGS.NUM_HIDDEN_UNITS])
        W_hidden = tf.get_variable('W_hidden', 
            [FLAGS.NUM_HIDDEN_UNITS, FLAGS.NUM_HIDDEN_UNITS])
        b_hidden = tf.get_variable('b_hidden', [FLAGS.NUM_HIDDEN_UNITS], 
            initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(rnn_input, W_input) + 
                   tf.matmul(state, W_hidden) + b_hidden)

def rnn_logits(FLAGS, rnn_output):
    with tf.variable_scope('softmax', reuse=True):
        W_softmax = tf.get_variable('W_softmax', 
            [FLAGS.NUM_HIDDEN_UNITS, FLAGS.NUM_CLASSES])
        b_softmax = tf.get_variable('b_softmax', 
            [FLAGS.NUM_CLASSES], initializer=tf.constant_initializer(0.0))
    return tf.matmul(rnn_output, W_softmax) + b_softmax

class model(object):

    def __init__(self, FLAGS):

        # Placeholders
        self.X = tf.placeholder(tf.int32, [None, None], 
            name='input_placeholder')
        self.y = tf.placeholder(tf.int32, [None, None], 
            name='labels_placeholder')
        self.initial_state = tf.zeros([FLAGS.NUM_BATCHES, FLAGS.NUM_HIDDEN_UNITS])

        # Prepre the inputs
        X_one_hot = tf.one_hot(self.X, FLAGS.NUM_CLASSES)
        rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) \
            for i in tf.split(1, FLAGS.SEQ_LEN, X_one_hot)]

        # Define the RNN cell
        with tf.variable_scope('rnn_cell'):
            W_input = tf.get_variable('W_input', 
                [FLAGS.NUM_CLASSES, FLAGS.NUM_HIDDEN_UNITS])
            W_hidden = tf.get_variable('W_hidden', 
                [FLAGS.NUM_HIDDEN_UNITS, FLAGS.NUM_HIDDEN_UNITS])
            b_hidden = tf.get_variable('b_hidden', 
                [FLAGS.NUM_HIDDEN_UNITS], 
                initializer=tf.constant_initializer(0.0))

        # Creating the RNN
        state = self.initial_state
        rnn_outputs = []
        for rnn_input in rnn_inputs:
            state = rnn_cell(FLAGS, rnn_input, state)
            rnn_outputs.append(state)
        self.final_state = rnn_outputs[-1]

        # Logits and predictions
        with tf.variable_scope('softmax'):
            W_softmax = tf.get_variable('W_softmax', 
                [FLAGS.NUM_HIDDEN_UNITS, FLAGS.NUM_CLASSES])
            b_softmax = tf.get_variable('b_softmax', 
                [FLAGS.NUM_CLASSES], 
                initializer=tf.constant_initializer(0.0))

        logits = [rnn_logits(FLAGS, rnn_output) for rnn_output in rnn_outputs]
        self.predictions = [tf.nn.softmax(logit) for logit in logits]

        # Loss and optimization
        y_as_list = [tf.squeeze(i, squeeze_dims=[1]) \
            for i in tf.split(1, FLAGS.SEQ_LEN, self.y)]
        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logit, label) \
            for logit, label in zip(logits, y_as_list)]
        self.total_loss = tf.reduce_mean(losses)
        self.train_step = tf.train.AdagradOptimizer(
            FLAGS.LEARNING_RATE).minimize(self.total_loss)

    def step(self, sess, batch_X, batch_y, initial_state):

        input_feed = {self.X: batch_X, 
                      self.y:batch_y, 
                      self.initial_state:initial_state}
        output_feed = [self.predictions,
                       self.total_loss,
                       self.final_state,
                       self.train_step]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2], outputs[3]

    def sample(self, FLAGS, sampling_type=1):

        initial_state = tf.zeros([1,FLAGS.NUM_HIDDEN_UNITS])
        predictions = []

        # Process preset tokens 
        state = initial_state
        for char in FLAGS.START_TOKEN:
            idx = FLAGS.char_to_idx[char]
            idx_one_hot = tf.one_hot(idx, FLAGS.NUM_CLASSES)
            rnn_input = tf.reshape(idx_one_hot, [1, 65])
            state =  rnn_cell(FLAGS, rnn_input, state)

        # Predict after preset tokens
        logit = rnn_logits(FLAGS, state)
        prediction = tf.argmax(tf.nn.softmax(logit), 1)[0]
        predictions.append(prediction.eval())

        for token_num in range(FLAGS.PREDICTION_LENGTH-1):
            idx_one_hot = tf.one_hot(prediction, FLAGS.NUM_CLASSES)
            rnn_input = tf.reshape(idx_one_hot, [1, 65])
            state =  rnn_cell(FLAGS, rnn_input, state)
            logit = rnn_logits(FLAGS, state)

            # scale the distribution
            # for creativity, higher temperatures produce more nonexistent words 
            # BUT more creative samples
            next_char_dist = logit/FLAGS.TEMPERATURE 
            next_char_dist = tf.exp(next_char_dist)
            next_char_dist /= tf.reduce_sum(next_char_dist)

            dist = next_char_dist.eval()

            # sample a character
            if sampling_type == 0:
                prediction = tf.argmax(tf.nn.softmax(
                                        next_char_dist), 1)[0].eval()
            elif sampling_type == 1:
                prediction = FLAGS.NUM_CLASSES - 1
                point = random.random()
                weight = 0.0
                for index in range(0, FLAGS.NUM_CLASSES):
                    weight += dist[0][index]
                    if weight >= point:
                        prediction = index
                        break
            else:
                raise ValueError("Pick a valid sampling_type!")
            predictions.append(prediction)

        return predictions

def create_model(sess, FLAGS):

    char_model = model(FLAGS)
    sess.run(tf.initialize_all_variables())
    return char_model

def train(FLAGS):

    # Start tensorflow session
    with tf.Session() as sess:

        model = create_model(sess, FLAGS)

        for idx, epoch in enumerate(generate_epochs(FLAGS)):

            training_losses = []
            state = np.zeros([FLAGS.NUM_BATCHES, FLAGS.NUM_HIDDEN_UNITS])

            for step, (input_X, input_y) in enumerate(epoch):
                predictions, total_loss, state, _= model.step(sess, input_X, input_y,
                                                        state)
                training_losses.append(total_loss)
            print "Epoch %i, loss: %.3f" % (idx, np.mean(training_losses))

            # Generate predictions
            if idx%10 == 0:
                print "Prediction:"
                predictions = model.sample(FLAGS)
                print FLAGS.START_TOKEN + "".join([FLAGS.idx_to_char[prediction]\
                    for prediction in predictions])
                print


    return training_losses

if __name__ == '__main__':
    FLAGS = parameters()
    
    data = open(FLAGS.FILE, "r").read()
    chars = list(set(data))
    char_to_idx = {char:i for i, char in enumerate(chars)}
    idx_to_char = {i:char for i, char in enumerate(chars)}

    FLAGS.data = data
    FLAGS.DATA_SIZE = len(data)
    FLAGS.NUM_CLASSES = len(chars)
    FLAGS.char_to_idx = char_to_idx
    FLAGS.idx_to_char = idx_to_char
    FLAGS.NUM_BATCHES = FLAGS.DATA_SIZE // FLAGS.BATCH_SIZE

    training_losses = train(FLAGS)

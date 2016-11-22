import tensorflow as tf

def rnn_cell(FLAGS, dropout):

    # Get the cell type
    if FLAGS.rnn_unit == 'rnn':
        rnn_cell_type = tf.nn.rnn_cell.BasicRNNCell
    elif FLAGS.rnn_unit == 'gru':
        rnn_cell_type = tf.nn.rnn_cell.GRUCell
    elif FLAGS.rnn_unit == 'lstm':
        rnn_cell_type = tf.nn.rnn_cell.BasicLSTMCell
    else:
        raise Exception("Choose a valid RNN unit type.")

    # Single cell
    single_cell = rnn_cell_type(FLAGS.num_hidden_units)

    # Dropout
    single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell,
        output_keep_prob=1-dropout)

    # Each state as one cell
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
        [single_cell] * FLAGS.num_layers)

    return stacked_cell

def rnn_inputs(FLAGS, input_data):

    with tf.variable_scope('rnn_inputs', reuse=True):
        W_input = tf.get_variable("W_input",
            [FLAGS.en_vocab_size, FLAGS.num_hidden_units])

    # <num_examples, seq_len, num_hidden_units>
    embeddings = tf.nn.embedding_lookup(W_input, input_data)

    return embeddings

def rnn_softmax(FLAGS, outputs):
    with tf.variable_scope('rnn_softmax', reuse=True):
        W_softmax = tf.get_variable("W_softmax",
            [FLAGS.num_hidden_units, FLAGS.num_classes])
        b_softmax = tf.get_variable("b_softmax", [FLAGS.num_classes])

    logits = tf.matmul(outputs, W_softmax) + b_softmax

    return logits

def length(data):
    relevant = tf.sign(tf.abs(data))
    length = tf.reduce_sum(relevant, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

class model(object):

    def __init__(self, FLAGS):

        # Placeholders
        self.inputs_X = tf.placeholder(tf.int32,
            shape=[None, None], name='inputs_X')
        self.targets_y = tf.placeholder(tf.float32,
            shape=[None, None], name='targets_y')
        self.seq_lens = tf.placeholder(tf.int32,
            shape=[None, ], name='seq_lens')
        self.dropout = tf.placeholder(tf.float32)

        # RNN cell
        stacked_cell = rnn_cell(FLAGS, self.dropout)

        # Inputs to RNN
        with tf.variable_scope('rnn_inputs'):
            W_input = tf.get_variable("W_input",
                [FLAGS.en_vocab_size, FLAGS.num_hidden_units])

        inputs = rnn_inputs(FLAGS, self.inputs_X)
        #initial_state = stacked_cell.zero_state(FLAGS.batch_size, tf.float32)

        # Outputs from RNN
        all_outputs, state = tf.nn.dynamic_rnn(cell=stacked_cell, inputs=inputs,
            sequence_length=self.seq_lens, dtype=tf.float32)

        # state has the last RELEVANT output automatically since we fed in seq_len
        # [0] because state is a tuple with a tensor inside it
        outputs = state[0]

        # Process RNN outputs
        with tf.variable_scope('rnn_softmax'):
            W_softmax = tf.get_variable("W_softmax",
                [FLAGS.num_hidden_units, FLAGS.num_classes])
            b_softmax = tf.get_variable("b_softmax", [FLAGS.num_classes])

        # Logits
        logits = rnn_softmax(FLAGS, outputs)
        probabilities = tf.nn.softmax(logits)
        self.accuracy = tf.equal(tf.argmax(
            self.targets_y,1), tf.argmax(logits,1))

        # Loss
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits, self.targets_y))

        # Optimization
        self.lr = tf.Variable(0.0, trainable=False)
        trainable_vars = tf.trainable_variables()
        # clip the gradient to avoid vanishing or blowing up gradients
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, trainable_vars), FLAGS.max_gradient_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_optimizer = optimizer.apply_gradients(
            zip(grads, trainable_vars))

        # Below are values we will use for sampling (generating the sentiment
        # after each word.)

        # this is taking all the ouputs for the first input sequence
        # (only 1 input sequence since we are sampling)
        sampling_outputs = all_outputs[0]

        # Logits
        sampling_logits = rnn_softmax(FLAGS, sampling_outputs)
        self.sampling_probabilities = tf.nn.softmax(sampling_logits)

        # Components for model saving
        self.global_step = tf.Variable(0, trainable=False)
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, sess, batch_X, batch_seq_lens, batch_y=None, dropout=0.0,
        forward_only=True, sampling=False):

        input_feed = {self.inputs_X: batch_X,
                      self.targets_y: batch_y,
                      self.seq_lens: batch_seq_lens,
                      self.dropout: dropout}

        if forward_only:
            if not sampling:
                output_feed = [self.loss,
                               self.accuracy]
            elif sampling:
                input_feed = {self.inputs_X: batch_X,
                              self.seq_lens: batch_seq_lens,
                              self.dropout: dropout}
                output_feed = [self.sampling_probabilities]
        else: # training
            output_feed = [self.train_optimizer,
                           self.loss,
                           self.accuracy]


        outputs = sess.run(output_feed, input_feed)

        if forward_only:
            if not sampling:
                return outputs[0], outputs[1]
            elif sampling:
                return outputs[0]
        else: # training
            return outputs[0], outputs[1], outputs[2]


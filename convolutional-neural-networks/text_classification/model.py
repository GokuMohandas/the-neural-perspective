import tensorflow as tf

def embed_inputs(FLAGS, input_data):

    with tf.variable_scope('embed_inputs', reuse=True):
        W_input = tf.get_variable("W_input",
            [FLAGS.en_vocab_size, FLAGS.num_hidden_units])

    # <num_examples, seq_len, num_hidden_units>
    embeddings = tf.nn.embedding_lookup(W_input, input_data)
    return embeddings

class model(object):

    def __init__(self, FLAGS):

        # Placeholders
        self.inputs_X = tf.placeholder(tf.int32,
            shape=[None, None], name='inputs_X')
        self.targets_y = tf.placeholder(tf.float32,
            shape=[None, None], name='targets_y')
        self.seq_lens = tf.placeholder(tf.int32,
            shape=[None, ], name='seq_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        # Inputs to RNN
        with tf.variable_scope('embed_inputs'):
            W_input = tf.get_variable("W_input",
                [FLAGS.en_vocab_size, FLAGS.num_hidden_units])
        self.embedded_inputs = embed_inputs(FLAGS, self.inputs_X)

        # Made embedded inputs in to 4D input for CNN
        self.conv_inputs = tf.expand_dims(self.embedded_inputs, -1)

        self.pooled_outputs = []
        for i, filter_size in enumerate(FLAGS.filter_sizes):
            with tf.name_scope("CNN-%s" % filter_size):
                # Convolution Layer
                filter_shape = [
                    filter_size, FLAGS.num_hidden_units, 1, FLAGS.num_filters]
                W = tf.Variable(
                    tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(
                    tf.constant(0.1, shape=[FLAGS.num_filters]), name="b")
                self.conv = tf.nn.conv2d(
                    input=self.conv_inputs,
                    filter=W,
                    strides=[1, 1, 1, 1], # S = 1
                    padding="VALID", # P = 0
                    name="conv")

                # Add bias and then apply the nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(self.conv, b))

                # apply max pool
                self.pooled = tf.nn.max_pool(
                    value=h,
                    ksize=[1, FLAGS.max_sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1], # usually [1, 2, 2, 1] for reduction
                    padding="VALID", # P = 0
                    name="pool")
                self.pooled_outputs.append(self.pooled)

        # Combine all the pooled features and flatten
        num_filters_total = FLAGS.num_filters * len(FLAGS.filter_sizes)
        self.h_pool = tf.concat(3, self.pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Apply dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Process CNN outputs
        with tf.variable_scope('rnn_softmax'):
            W_softmax = tf.get_variable("W_softmax",
                [num_filters_total, FLAGS.num_classes])
            b_softmax = tf.get_variable("b_softmax", [FLAGS.num_classes])

        # Logits
        logits = tf.matmul(self.h_drop, W_softmax) + b_softmax
        self.probabilities = tf.nn.softmax(logits)
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

        # Components for model saving
        self.global_step = tf.Variable(0, trainable=False)
        self.saver = tf.train.Saver(tf.all_variables())


    def step(self, sess, batch_X, batch_seq_lens, batch_y=None,
        dropout_keep_prob=1.0, forward_only=True, sampling=False):

        input_feed = {self.inputs_X: batch_X,
                      self.targets_y: batch_y,
                      self.seq_lens: batch_seq_lens,
                      self.dropout_keep_prob: dropout_keep_prob}

        if forward_only:
            if not sampling:
                output_feed = [self.loss,
                               self.accuracy]
            elif sampling:
                input_feed = {self.inputs_X: batch_X,
                              self.seq_lens: batch_seq_lens,
                              self.dropout_keep_prob: dropout_keep_prob}
                output_feed = [self.probabilities]
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


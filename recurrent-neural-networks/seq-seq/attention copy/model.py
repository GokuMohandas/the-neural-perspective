import tensorflow as tf
from tensorflow.python.ops import (
    seq2seq,
)

def rnn_cell(FLAGS, dropout, scope):

    with tf.variable_scope(scope):
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

def rnn_inputs(FLAGS, input_data, vocab_size, scope):

    with tf.variable_scope(scope, reuse=True):
        W_input = tf.get_variable("W_input",
            [vocab_size, FLAGS.num_hidden_units])

        # embeddings will be shape [input_data dimensions, num_hidden units]
        embeddings = tf.nn.embedding_lookup(W_input, input_data)
        return embeddings

def rnn_softmax(FLAGS, outputs, scope):
    with tf.variable_scope(scope, reuse=True):
        W_softmax = tf.get_variable("W_softmax",
            [FLAGS.num_hidden_units, FLAGS.sp_vocab_size])
        b_softmax = tf.get_variable("b_softmax", [FLAGS.sp_vocab_size])

        logits = tf.matmul(outputs, W_softmax) + b_softmax
        return logits

class model(object):

    def __init__(self, FLAGS, forward_only):

        # Placeholders
        self.encoder_inputs = tf.placeholder(tf.int32,
            shape=[FLAGS.batch_size, None],
            name='encoder_inputs')
        self.decoder_inputs = tf.placeholder(tf.int32,
            shape=[FLAGS.batch_size, FLAGS.sp_max_len],
            name='decoder_inputs')
        self.targets = tf.placeholder(tf.int32,
            shape=[FLAGS.batch_size, None],
            name='targets')
        self.en_seq_lens = tf.placeholder(tf.int32,
            shape=[FLAGS.batch_size, ],
            name="en_seq_lens")
        self.sp_seq_lens = tf.placeholder(tf.int32,
            shape=[FLAGS.batch_size, ],
            name="sp_seq_lens")
        self.dropout = tf.placeholder(tf.float32)

        with tf.variable_scope('encoder') as scope:

            # Encoder RNN cell
            self.encoder_stacked_cell = rnn_cell(FLAGS, self.dropout,
                scope=scope)

            # Embed encoder inputs
            W_input = tf.get_variable("W_input",
                [FLAGS.en_vocab_size, FLAGS.num_hidden_units])
            self.embedded_encoder_inputs = rnn_inputs(FLAGS,
                self.encoder_inputs, FLAGS.en_vocab_size, scope=scope)
            #initial_state = encoder_stacked_cell.zero_state(FLAGS.batch_size, tf.float32)

            # Outputs from encoder RNN
            self.all_encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                cell=self.encoder_stacked_cell,
                inputs=self.embedded_encoder_inputs,
                sequence_length=self.en_seq_lens, time_major=False,
                dtype=tf.float32)


        with tf.variable_scope('attention') as scope:
            # Convert to list of tensors
            self.list_encoder_outputs = tf.unpack(
                self.all_encoder_outputs, axis=0)
            self.list_encoder_state = tf.unpack(
                self.encoder_state, axis=0)

            # First calculate a concatenation of encoder outputs to put attention on.
            #self.top_states = [tf.reshape(e, [
            #    -1, 1, self.encoder_stacked_cell.output_size]) \
            #        for e in self.list_encoder_outputs]

            # BELOW does the following transformation:
            # (batch_size, max_len, 1, num_hidden_units) =
            #   bath_size list of (max_len, 1, num_hidden_units)
            # ---> (max_len, batch_size, num_hidden_units)
            #self.attention_states = tf.concat(1, self.top_states)

            # Need attention states to be [batch_size, attn_length, attn_size]
            self.attention_states = self.all_encoder_outputs


        with tf.variable_scope('decoder') as scope:

            # Initial state is last relevant state from encoder
            self.decoder_initial_state = self.encoder_state

            # Decoder RNN cell
            self.decoder_stacked_cell = rnn_cell(FLAGS, self.dropout,
                scope=scope)

            # Embed decoder RNN inputs
            #W_input = tf.get_variable("W_input",
            #    [FLAGS.sp_vocab_size, FLAGS.num_hidden_units])
            #self.embedded_decoder_inputs = rnn_inputs(FLAGS, self.decoder_inputs,
            #    FLAGS.sp_vocab_size, scope=scope)

            self.decoder_inputs_time_major = tf.transpose(
                self.decoder_inputs, [1, 0])

            # make decoder inputs into a batch_size list of inputs
            self.list_decoder_inputs = tf.unpack(
                self.decoder_inputs_time_major, axis=0)

            # Output project weights (fot softmax and embedding predictions)
            W_softmax = tf.get_variable("W_softmax",
                shape=[FLAGS.num_hidden_units, FLAGS.sp_vocab_size],
                dtype=tf.float32)
            b_softmax = tf.get_variable("b_softmax",
                shape=[FLAGS.sp_vocab_size],
                dtype=tf.float32)
            output_projection = (W_softmax, b_softmax)

            self.all_decoder_outputs, self.decoder_state = \
                seq2seq.embedding_attention_decoder(
                    decoder_inputs=self.list_decoder_inputs,
                    initial_state=self.decoder_initial_state,
                    attention_states=self.attention_states,
                    cell=self.decoder_stacked_cell,
                    num_symbols=FLAGS.sp_vocab_size,
                    embedding_size=FLAGS.num_hidden_units,
                    output_projection=output_projection,
                    feed_previous=forward_only)

            '''
            # Outputs from encoder RNN
            self.all_decoder_outputs, self.decoder_state = tf.nn.dynamic_rnn(
                cell=self.decoder_stacked_cell,
                inputs=self.embedded_decoder_inputs,
                sequence_length=self.sp_seq_lens, time_major=False,
                initial_state=self.decoder_initial_state)
            '''

            # Logits
            self.decoder_outputs_flat = tf.reshape(self.all_decoder_outputs,
                [-1, FLAGS.num_hidden_units])
            self.logits_flat = rnn_softmax(FLAGS, self.decoder_outputs_flat,
                scope=scope)

            # Loss with masking
            targets_flat = tf.reshape(self.targets, [-1])
            losses_flat = tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.logits_flat, targets_flat)
            mask = tf.sign(tf.to_float(targets_flat))
            masked_losses = mask * losses_flat
            masked_losses = tf.reshape(masked_losses,  tf.shape(self.targets))
            self.loss = tf.reduce_mean(
                tf.reduce_sum(masked_losses, reduction_indices=1))

        # Optimization
        self.lr = tf.Variable(0.0, trainable=False)
        trainable_vars = tf.trainable_variables()
        # clip the gradient to avoid vanishing or blowing up gradients
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, trainable_vars), FLAGS.max_gradient_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_optimizer = optimizer.apply_gradients(
            zip(grads, trainable_vars))

        # For printng results from training
        self.logits = tf.reshape(self.logits_flat, [FLAGS.batch_size,
            FLAGS.sp_max_len, FLAGS.sp_vocab_size])
        self.y_pred = tf.argmax(self.logits, 2)

        # Save all the variables
        self.global_step = tf.Variable(0, trainable=False)
        self.saver = tf.train.Saver(tf.all_variables())


    def step(self, sess, FLAGS, batch_encoder_inputs, batch_decoder_inputs,
        batch_targets, batch_en_seq_lens, batch_sp_seq_lens, dropout,
        forward_only, sampling=False):

        if not forward_only and not sampling:
            input_feed = {self.encoder_inputs: batch_encoder_inputs,
                self.decoder_inputs: batch_decoder_inputs,
                self.targets: batch_targets,
                self.en_seq_lens: batch_en_seq_lens,
                self.sp_seq_lens: batch_sp_seq_lens,
                self.dropout: dropout}
            #output_feed = [self.loss, self.train_optimizer]
            output_feed = [self.y_pred, self.loss, self.train_optimizer]
            outputs = sess.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]
        elif forward_only and not sampling:
            input_feed = {self.encoder_inputs: batch_encoder_inputs,
                self.decoder_inputs: batch_decoder_inputs,
                self.targets: batch_targets,
                self.en_seq_lens: batch_en_seq_lens,
                self.sp_seq_lens: batch_sp_seq_lens,
                self.dropout: dropout}
            output_feed = [self.loss]
            outputs = sess.run(output_feed, input_feed)
            return outputs[0]






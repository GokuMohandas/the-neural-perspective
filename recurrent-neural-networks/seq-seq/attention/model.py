import tensorflow as tf
from tensorflow.python.ops import (
    seq2seq,
)
from tensorflow.python.util import (
    nest,
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

def _extract_argmax_and_embed(W_embedding, output_projection,
    update_embedding=True):
    """
    Extract the argmax from the decoder outputs and use the output_projection
    weights to project the predicted output as an input to the next
    decoder cell. update_embedding will either allow or freeze W_embedding.

    Return will be a loop function.
    """

    def loop_function(prev, _):
        """
        prev is the previous decoder output. _ is just a placeholder
        for something like an index i.
        """

        # xW + b to convert decoder ouput [N, H] to shape [N, C]
        ''' Recall that decoder inputs are time-major so the output from any one
            cell is [N, H] '''
        prev = tf.matmul(prev, output_projection[0]) + output_projection[1]

        # Extract argmax
        prev_symbol = tf.argmax(prev, dimension=1)

        # Need to embed the prev_symbol before feeding into next decoder cell
        embedded_prev_symbol = tf.nn.embedding_lookup(W_embedding, prev_symbol)

        # Stop the gradient update is update_embedding is False
        ''' This mean the embedding the output projection will not alter the
            decoder input embeddings (W_embedding). '''
        if not update_embedding:
            embedded_prev_symbol = tf.stop_gradient(embedded_prev_symbol)

        return embedded_prev_symbol

    return loop_function

def attention_decoder(decoder_inputs, initial_state, attention_states,
    cell, output_size, num_heads=1, loop_function=None, dtype=None,
    scope=None, initial_state_attention=False):
    """
    The decoder with an attentional interface.
    """

    # Set the scope
    with tf.variable_scope(scope or 'attention_decoder', dtype=dtype) as scope:

        # set the dtype
        dtype = scope.dtype

        # Get sizes
        batch_size = tf.shape(decoder_inputs[0])[0] # decoder_inputs is a list
        attn_length = attention_states.get_shape()[1].value # [N, <max_len> H]
        if attn_length == None: # encoder inputs placeholder had None for 2nd D
            attn_length = tf.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # Use a 1x1 convolution to process encoder hidden states into features
        ''' This is an additional step we use (not in the paper). Instead of
            using attention on the raw hidden state outputs from the encoder,
            we use convolution to extract meaningful features from the hidden
            states. We will then apply attention on our ouputs from the conv.

            Recall, to calculate e_ij we need to feed in our encoder hidden
            states and the previous hidden state from the decoder into a tanh.
            We could feed in the raw encoder hidden states and previous decoder
            hidden state in raw into the tanh and apply the nonlinearity weights
            and then take the tanh of that. But, instead, we split it up. The
            nonlinearity is applied to the encoder hidden states via this conv
            operations and the previous decoder hidden state (query) goes
            through its own separate weights with _linear. Finally both results
            are added and then the regular tanh is applied. We multiply this
            value by v (which acts as softmax weights)

            Transformation in shape:
                original hidden state:
                    [N, max_len, H]
                reshaped to 4D hidden:
                    [N, max_len, 1, H] = N images of [max_len, 1, H]
                    so we can apply filter
                filter:
                    [1, 1, H, H] = [height, width, depth, # num filters]
                Apply conv with stride 1 and padding 1:
                    H = ((H - F + 2P) / S) + 1 =
                        ((max_len - 1 + 2)/1) + 1 = height'
                    W = ((W - F + 2P) / S) + 1 = ((1 - 1 + 2)/1) + 1 = 3
                    K = K = H
                    So we just converted a
                        [N, max_len, H] into [N, height', 3, H]

        '''
        hidden = tf.reshape(attention_states,
            [-1, attn_length, 1, attn_size]) # [N, max_len, 1, H]
        hidden_features = []
        attention_softmax_weights = []
        for a in xrange(num_heads):
            # filter
            k = tf.get_variable("AttnW_%d" % a,
                [1, 1, attn_size, attn_size]) # [1, 1, H, H]
            hidden_features.append(tf.nn.conv2d(hidden, k, [1,1,1,1], "SAME"))
            attention_softmax_weights.append(tf.get_variable(
                "W_attention_softmax_%d" % a, [attn_size]))

        state = initial_state

        def attention(query):
            """
            Places an attention mask on hidden states from encoder
            using hidden and query. Query is a state of shape [N, H]
            """
            # results of the attention reads
            cs = [] # context vectors c_i

            # Flatten the query if it is a tuple
            if nest.is_sequence(query):
                # converts query from [N, H] to list of size N if [H, 1]
                query_list = nest.flatten(query)
            query = tf.concat(1, query_list) # becomes [H, N]

            for a in xrange(num_heads):
                with tf.variable_scope("Attention_%d" % a) as scope:
                    y = tf.nn.rnn_cell._linear(
                        args=query, output_size=attn_size, bias=True)

                    # Reshape into 4D
                    y = tf.reshape(y, [-1, 1, 1, attn_size]) # [N, 1, 1, H]

                    # Calculating alpha
                    s = tf.reduce_sum(
                        attention_softmax_weights[a] *
                        tf.nn.tanh(hidden_features[a] + y), [2, 3])
                    a = tf.nn.softmax(s)

                    # Calculate context c
                    c = tf.reduce_sum(tf.reshape(
                        a, [-1, attn_length, 1, 1])*hidden, [1,2])
                    cs.append(tf.reshape(c, [-1, attn_size]))

            return cs

        outputs = []
        prev = None
        batch_attn_size = tf.pack([batch_size, attn_size])
        attns = [tf.zeros(batch_attn_size, dtype=dtype)
            for _ in xrange(num_heads)]
        for a in attns:
            a.set_shape([None, attn_size])

        # Process decoder inputs one by one
        for i, inp in enumerate(decoder_inputs):

            if i > 0:
                tf.get_variable_scope().reuse_variables()

            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)

            # Merge the input and attentions together
            input_size = inp.get_shape().with_rank(2)[1]
            x = tf.nn.rnn_cell._linear(
                args=[inp]+attns, output_size=input_size, bias=True)

            # Decoder RNN
            cell_outputs, state = cell(x, state) # our stacked cell

            # Attention mechanism to get Cs
            attns = attention(state)

            with tf.variable_scope('attention_output_projection'):
                output = tf.nn.rnn_cell._linear(
                    args=[cell_outputs]+attns, output_size=output_size,
                    bias=True)
            if loop_function is not None:
                prev = output
            outputs.append(output)

        return outputs, state

def embedding_attention_decoder(decoder_inputs, initial_state,
    attention_states, cell, num_symbols, embedding_size, output_projection,
    feed_previous, num_heads=1, update_embedding_for_previous=True,
    scope=None, dtype=None, initial_state_attention=False):
    """
    Embedding attention decoder which will produce the decoder outputs
    and state. Feed previous will be used to determine wether to feed in the
    previous decoder output into consideration for the next prediction or not.

    Args:
    decoder_inputs: <list> in time-major [<max_len>, N],
    initial_state: [N, H],
    attention_states: [batch_size, attn_length, attn_size] = [N, <max_len> H],
    cell: decoder layered cell,
    num_symbols: sp vocab size,
    embedding_size: embedding size (usually just H),
    output_projection: (W, b) for softmax and projection of decoder outputs
    feed_previous: boolean (True if feeding in previous decoder output)
    num_heads: number of attention heads (usually just 1)
    update_embedding_for_previous: boolean, if feed_previous is False, this
        variable has no effect. If feed_previous is True and this is True,
        then the softmax/output_projection (which are same set of weights)
        weights will be updated when using the decoder output, embedding it,
        and feeding into the next decoder rnn cell to use for aid in prediction
        of the next decoder output.  If feed_previous is True and this is False,
        the weights will not be altered when we do the output_projection except
        for the GO token's embedding weights.
    initial_state_attention: if you want initialize attentions (False).
    dtype: None defaults to tf.float32

    Return:
    outputs: [<max_len>, N, H]
    state: [N, H]

    """

    # Get the output dimension from the cell
    output_size = cell.output_size

    # Set up scope
    with tf.variable_scope(scope or "embedding_attention_decoder",
        dtype=dtype) as scope:

        # Embed the decoder inputs (which is a list)
        W_embedding = tf.get_variable("W_embedding",
            shape=[num_symbols, embedding_size])
        embedded_decoder_inputs = [
            tf.nn.embedding_lookup(W_embedding, i) for i in decoder_inputs]

        # Loop function if using decoder outputs for next prediction
        loop_function = _extract_argmax_and_embed(
            W_embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None

        return attention_decoder(
            embedded_decoder_inputs,
            initial_state,
            attention_states,
            cell,
            output_size=output_size,
            num_heads=num_heads,
            loop_function=loop_function,
            initial_state_attention=initial_state_attention)

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
            # Need attention states to be [batch_size, attn_length, attn_size]
            self.attention_states = self.all_encoder_outputs


        with tf.variable_scope('decoder') as scope:

            # Initial state is last relevant state from encoder
            self.decoder_initial_state = self.encoder_state

            # Decoder RNN cell
            self.decoder_stacked_cell = rnn_cell(FLAGS, self.dropout,
                scope=scope)

            # Need input to embedding_attention_decoder to be time-major
            self.decoder_inputs_time_major = tf.transpose(
                self.decoder_inputs, [1, 0])

            # make decoder inputs into a batch_size list of inputs
            self.list_decoder_inputs = tf.unpack(
                self.decoder_inputs_time_major, axis=0)

            # Output projection weights (for softmax and embedding predictions)
            W_softmax = tf.get_variable("W_softmax",
                shape=[FLAGS.num_hidden_units, FLAGS.sp_vocab_size],
                dtype=tf.float32)
            b_softmax = tf.get_variable("b_softmax",
                shape=[FLAGS.sp_vocab_size],
                dtype=tf.float32)
            output_projection = (W_softmax, b_softmax)

            self.all_decoder_outputs, self.decoder_state = \
                embedding_attention_decoder(
                    decoder_inputs=self.list_decoder_inputs,
                    initial_state=self.decoder_initial_state,
                    attention_states=self.attention_states,
                    cell=self.decoder_stacked_cell,
                    num_symbols=FLAGS.sp_vocab_size,
                    embedding_size=FLAGS.num_hidden_units,
                    output_projection=output_projection,
                    feed_previous=forward_only)

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
            input_feed = {
                self.encoder_inputs: batch_encoder_inputs,
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
            input_feed = {
                self.encoder_inputs: batch_encoder_inputs,
                self.decoder_inputs: batch_decoder_inputs,
                self.targets: batch_targets,
                self.en_seq_lens: batch_en_seq_lens,
                self.sp_seq_lens: batch_sp_seq_lens,
                self.dropout: dropout}
            output_feed = [self.loss]
            outputs = sess.run(output_feed, input_feed)
            return outputs[0]
        elif forward_only and sampling:
            input_feed = {
                self.encoder_inputs: batch_encoder_inputs,
                self.decoder_inputs: batch_decoder_inputs,
                self.en_seq_lens: batch_en_seq_lens,
                self.dropout: dropout}
            output_feed = [self.y_pred]
            outputs = sess.run(output_feed, input_feed)
            return outputs[0]







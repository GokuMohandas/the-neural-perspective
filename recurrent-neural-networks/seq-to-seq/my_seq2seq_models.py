import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

def rnn_inputs(source_vocab_size, embedding_size, input_data):

    with tf.variable_scope('rnn_inputs', reuse=True):
        W_input = tf.get_variable("W_input", [source_vocab_size, embedding_size])

    # <batch_size, seq_len, num_hidden_units>
    # embedding_loopup one hot encodes our raw input and then embeds it
    embeddings = tf.nn.embedding_lookup(W_input, input_data)

    return embeddings

def length(data):
    relevant = tf.sign(tf.abs(data))

    # 0 if data is [max_time_steps, batch_size, depth]
    # 1 if data is [batch_size, max_time_steps, depth]
    length = tf.reduce_sum(relevant, reduction_indices=0)
    length = tf.cast(length, tf.int32)
    return length

def embedding_attention_seq2seq_with_biRNN_encoder(encoder_inputs,
                                        decoder_inputs,
                                        cell,
                                        num_encoder_symbols,
                                        num_decoder_symbols,
                                        embedding_size,
                                        num_heads=1,
                                        output_projection=None,
                                        feed_previous=False,
                                        dtype=None,
                                        scope=None,
                                        initial_state_attention=False):

    """Embedding sequence-to-sequence model with attention.
    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. It keeps the outputs of this
    RNN at every step to use for attention later. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs attention decoder, initialized with the last
    encoder state, on embedded decoder_inputs and attending to encoder outputs.

    Args:
        encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        num_encoder_symbols: Integer; number of symbols on the encoder side.
        num_decoder_symbols: Integer; number of symbols on the decoder side.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_decoder_symbols] and B has
        shape [num_decoder_symbols]; if provided and feed_previous=True, each
        fed previous output will first be multiplied by W and added B.
        feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
        of decoder_inputs will be used (the "GO" symbol), and all other decoder
        inputs will be taken from previous outputs (as in embedding_rnn_decoder).
        If False, decoder_inputs are used as given (the standard decoder case).
        dtype: The dtype of the initial RNN state (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
        "embedding_attention_seq2seq".
        initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states.

    Returns:
        A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
        state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
    """

    with variable_scope.variable_scope(
        scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype

        # Embed the inputs
        with tf.variable_scope('rnn_inputs'):
            W_input = tf.get_variable("W_input", [num_encoder_symbols, embedding_size])

        #batch_size = len(encoder_inputs)
        inputs = rnn_inputs(num_encoder_symbols, embedding_size, encoder_inputs)
        #initial_state = cell.zero_state(batch_size, tf.float32)

        # Outputs from encoder RNN
        seq_lens = length(encoder_inputs)

        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell,
            inputs=inputs, initial_state_fw=None, initial_state_bw=None,
            sequence_length=seq_lens, time_major=True, dtype=tf.float32)
        outputs_fw, outputs_bw = outputs
        state_fw, state_bw = state

        encoder_outputs = tf.add(outputs_fw, outputs_bw)
        encoder_state = tf.add(state_fw, state_bw)

        # Convert to list of tensors which is what rest of functions below need
        encoder_outputs = tf.unpack(encoder_outputs, axis=0) # annotations
        encoder_state = tf.unpack(encoder_state, axis=0)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs]
        attention_states = array_ops.concat(1, top_states)

        # Decoder.
        output_size = None
        if output_projection is None:
            cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        if isinstance(feed_previous, bool):
            return seq2seq.embedding_attention_decoder(
                decoder_inputs,
                encoder_state,
                attention_states,
                cell,
                num_decoder_symbols,
                embedding_size,
                num_heads=num_heads,
                output_size=output_size,
                output_projection=output_projection,
                feed_previous=feed_previous,
                initial_state_attention=initial_state_attention)


'''
def encompassing_model(encoder_inputs, decoder_inputs, targets, weights,
                        buckets, seq2seq, softmax_loss_function=None,
                        per_example_loss=False, name=None):

    with variable_scope.variable_scope("encompassing_model") as scope:

        outputs, _ = seq2seq(encoder_inputs, decoder_inputs)

        if per_example_loss:
            losses.append(sequence_loss_by_example(
                outputs, targets, weights,
                softmax_loss_function=softmax_loss_function))
        else:
            losses.append(sequence_loss(
                outputs, targets, weights,
                softmax_loss_function=softmax_loss_function))

    return outputs, losses
'''


# LN funcition
def ln(inputs, epsilon = 1e-5, scope = None):

    """ Computer LN given an input tensor. We get in an input of shape
    [N X D] and with LN we compute the mean and var for each individual
    training point across all it's hidden dimensions rather than across
    the training batch as we do in BN. This gives us a mean and var of shape
    [N X 1].
    """
    mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
    with tf.variable_scope(scope + 'LN'):
        scale = tf.get_variable('alpha',
                                shape=[inputs.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('beta',
                                shape=[inputs.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift

    return LN


# Modified from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py
class GRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = array_ops.split(1, 2, _linear([inputs, state],
                                             2 * self._num_units, True, 1.0))

        # Apply Layer Normalization to the two gates
        r = ln(r, scope = 'r/')
        u = ln(r, scope = 'u/')

        r, u = sigmoid(r), sigmoid(u)
      with vs.variable_scope("Candidate"):
        c = self._activation(_linear([inputs, r * state],
                                     self._num_units, True))
      new_h = u * state + (1 - u) * c
    return new_h, new_h
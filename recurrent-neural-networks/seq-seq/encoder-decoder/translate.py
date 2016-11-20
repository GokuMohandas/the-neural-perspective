import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import (
    process_data,
    split_data,
    generate_epoch,
    generate_batch,
)

from model import (
    model,
)

class parameters(object):

    def __init__(self):
        """
        Holds all the parameters for NMT.
        """

        self.max_en_vocab_size = 5000
        self.max_sp_vocab_size = 5000

        self.num_epochs = 100
        self.batch_size = 4

        self.rnn_unit = 'gru'
        self.num_hidden_units = 300
        self.num_layers = 1
        self.dropout = 0.5
        self.learning_rate = 1e-3
        self.learning_rate_decay_factor = 0.99
        self.max_gradient_norm = 5.0

def create_model(sess, FLAGS):

    tf_model = model(FLAGS)
    print "Created a new model"
    sess.run(tf.initialize_all_variables())

    return tf_model

def train(FLAGS):

    # Load the data
    en_token_ids, en_seq_lens, en_vocab_dict, en_rev_vocab_dict = \
        process_data('data/en.p', max_vocab_size=5000, target_lang=False)
    sp_token_ids, sp_seq_lens, sp_vocab_dict, sp_rev_vocab_dict = \
        process_data('data/sp.p', max_vocab_size=5000, target_lang=True)

    # Split into train and validation sets
    train_encoder_inputs, train_decoder_inputs, train_targets, \
        train_en_seq_lens, train_sp_seq_len, \
        valid_encoder_inputs, valid_decoder_inputs, valid_targets, \
        valid_en_seq_lens, valid_sp_seq_len = \
        split_data(en_token_ids, sp_token_ids, en_seq_lens, sp_seq_lens,
            train_ratio=0.8)

    # Update parameters
    FLAGS.en_vocab_size = len(en_vocab_dict)
    FLAGS.sp_vocab_size = len(sp_vocab_dict)

    # Start session
    with tf.Session() as sess:

        # Create new model or load old one
        model = create_model(sess, FLAGS)

        # Training begins
        losses = []
        for epoch_num, epoch in enumerate(generate_epoch(train_encoder_inputs,
            train_decoder_inputs, train_targets,
            train_en_seq_lens, train_sp_seq_len,
            FLAGS.num_epochs, FLAGS.batch_size)):

            print "EPOCH: %i" % (epoch_num)
            # Decay learning rate
            sess.run(tf.assign(model.lr, FLAGS.learning_rate * \
                (FLAGS.learning_rate_decay_factor ** epoch_num)))

            batch_loss = []

            for batch_num, (batch_encoder_inputs, batch_decoder_inputs,
                batch_targets, batch_en_seq_lens,
                batch_sp_seq_lens) in enumerate(epoch):

                loss, _ = model.step(sess, FLAGS,
                    batch_encoder_inputs, batch_decoder_inputs, batch_targets,
                    batch_en_seq_lens, batch_sp_seq_lens,
                    FLAGS.dropout)

                batch_loss.append(loss)

            losses.append(np.mean(batch_loss))

        plt.plot(losses, label='loss')
        plt.legend()
        plt.show()












if __name__ == '__main__':

    FLAGS = parameters()
    train(FLAGS)
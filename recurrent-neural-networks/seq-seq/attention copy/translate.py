import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
        self.ckpt_dir = 'checkpoints/'

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

def create_model(sess, FLAGS, forward_only):


    tf_model = model(FLAGS, forward_only)

    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Restoring old model parameters from %s" %
                             ckpt.model_checkpoint_path)
        tf_model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created new model.")
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
    FLAGS.sp_max_len = max(sp_seq_lens) + 1 # GO token

    # Start session
    with tf.Session() as sess:

        # Create new model or load old one
        model = create_model(sess, FLAGS, forward_only=False)

        # Training begins
        train_losses = []
        valid_losses = []
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

                y_pred, loss, _ = model.step(sess, FLAGS,
                    batch_encoder_inputs, batch_decoder_inputs, batch_targets,
                    batch_en_seq_lens, batch_sp_seq_lens,
                    FLAGS.dropout, forward_only=False)

                batch_loss.append(loss)
            train_losses.append(np.mean(batch_loss))

            for valid_epoch_num, valid_epoch in enumerate(generate_epoch(valid_encoder_inputs,
                valid_decoder_inputs, valid_targets,
                valid_en_seq_lens, valid_sp_seq_len,
                num_epochs=1, batch_size=FLAGS.batch_size)):

                batch_loss = []

                for batch_num, (batch_encoder_inputs, batch_decoder_inputs,
                    batch_targets, batch_en_seq_lens,
                    batch_sp_seq_lens) in enumerate(valid_epoch):

                    loss = model.step(sess, FLAGS,
                        batch_encoder_inputs, batch_decoder_inputs, batch_targets,
                        batch_en_seq_lens, batch_sp_seq_lens,
                        FLAGS.dropout, forward_only=True, sampling=False)

                    batch_loss.append(loss)
                valid_losses.append(np.mean(batch_loss))

        # Save checkpoint.
        if not os.path.isdir(FLAGS.ckpt_dir):
            os.makedirs(FLAGS.ckpt_dir)
        checkpoint_path = os.path.join(FLAGS.ckpt_dir, "model.ckpt")
        print "Saving the model."
        model.saver.save(sess, checkpoint_path,
                         global_step=model.global_step)

        plt.plot(train_losses, label='train_loss')
        plt.plot(valid_losses, label='valid_loss')
        plt.legend()
        plt.show()

def sample(FLAGS):

    with tf.Session() as sess:

        # Load trained model
        model = create_model(sess, FLAGS, forward_only=True)

        # Change FLAGS parameters
        FLAGS.batch_size = 1




if __name__ == '__main__':

    FLAGS = parameters()
    train(FLAGS)
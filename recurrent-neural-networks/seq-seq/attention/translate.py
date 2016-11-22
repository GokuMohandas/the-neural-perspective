import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from data_utils import (
    GO_ID,
    basic_tokenizer,
    data_to_token_ids,
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
                        dropout=0.0, forward_only=True, sampling=False)

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

    # Load the data needed to convert your sentence
    en_token_ids, en_seq_lens, en_vocab_dict, en_rev_vocab_dict = \
        process_data('data/en.p', max_vocab_size=5000, target_lang=False)
    sp_token_ids, sp_seq_lens, sp_vocab_dict, sp_rev_vocab_dict = \
        process_data('data/sp.p', max_vocab_size=5000, target_lang=True)

    # Change FLAGS parameters
    FLAGS.batch_size = 1
    FLAGS.en_vocab_size = len(en_vocab_dict)
    FLAGS.sp_vocab_size = len(sp_vocab_dict)
    FLAGS.sp_max_len = max(sp_seq_lens) + 1 # GO token

    # Process sample sentence
    inference_sentence = ["I like to play tennis and eat sandwiches."]
    # Split into tokens
    tokenized = []
    for i in xrange(len(inference_sentence)):
        tokenized.append(basic_tokenizer(inference_sentence[i]))
    # Convert data to token ids
    data_as_tokens, sample_en_seq_lens = data_to_token_ids(
        tokenized, en_vocab_dict, target_lang=False, normalize_digits=True)

    # make dummy_sp_inputs
    dummy_sp_inputs = np.array([[GO_ID]*FLAGS.sp_max_len])
    sample_sp_seq_lens = np.array([len(dummy_sp_inputs)])

    print data_as_tokens
    print sample_en_seq_lens
    print dummy_sp_inputs
    print sample_sp_seq_lens

    with tf.Session() as sess:

        # Load trained model
        model = create_model(sess, FLAGS, forward_only=True)

        y_pred = model.step(sess, FLAGS, batch_encoder_inputs=data_as_tokens,
            batch_decoder_inputs=dummy_sp_inputs, batch_targets=None,
            batch_en_seq_lens=sample_en_seq_lens,
            batch_sp_seq_lens=sample_sp_seq_lens,
            dropout=0.0, forward_only=True, sampling=True)

        # compose the predicted sp sentence
        sp_sentence = []
        for idx in y_pred[0]:
            sp_sentence.append(sp_rev_vocab_dict[idx])
        print " ".join([word for word in sp_sentence])


if __name__ == '__main__':

    FLAGS = parameters()

    if sys.argv[1] == 'train':
        train(FLAGS)
    elif sys.argv[1] == 'sample':
        sample(FLAGS)
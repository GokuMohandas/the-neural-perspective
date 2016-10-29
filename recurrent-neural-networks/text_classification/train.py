from data_utils import *
from model import *

import tensorflow as tf
import os

# Configs
tf.app.flags.DEFINE_string("rnn_unit", 'lstm', 
					"Type of RNN unit: rnn|gru|lstm.")

tf.app.flags.DEFINE_float("learning_rate", 1e-5, 
					"Learning rate.")

tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
					"Learning rate decays by this much.")

tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
					"Clip gradients to this norm.")

tf.app.flags.DEFINE_integer("num_epochs", 100,
					"Number of epochs during training.")

tf.app.flags.DEFINE_integer("batch_size", 64,
					"Batch size to use during training.")

tf.app.flags.DEFINE_integer("num_hidden_units", 300, 
					"Number of hidden units in each RNN unit.")

tf.app.flags.DEFINE_integer("num_layers", 1, 
					"Number of layers in the model.")

tf.app.flags.DEFINE_float("dropout", 0.5, 
					"Amount to drop during training.")

tf.app.flags.DEFINE_integer("en_vocab_size", 5000, 
					"English vocabulary size.")

tf.app.flags.DEFINE_string("ckpt_dir", "checkpoints",
					"Directory to save the model checkpoints")

tf.app.flags.DEFINE_integer("num_classes", 2,
					"Number of classification classes.")


FLAGS = tf.app.flags.FLAGS

X, y = load_data_and_labels()
vocab_list, vocab_dict, rev_vocab_dict = create_vocabulary(X)
X, seq_lens = data_to_token_ids(X, vocab_dict)
train_X, train_y, train_seq_lens, valid_X, valid_y, valid_seq_lens = split_data(X, y, seq_lens)


if __name__ == '__main__':

	g = model(FLAGS, train=True)

	# Variables for saving the model along the training procedure
	if not os.path.exists(FLAGS.ckpt_dir):
		os.makedirs(FLAGS.ckpt_dir)

	with tf.Session() as sess:
		tf.initialize_all_variables().run()

		# Load old model state if available 
		saver = tf.train.Saver(tf.all_variables())
		ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
		if ckpt and ckpt.model_checkpoint_path:
			print "Loading old model from:", ckpt.model_checkpoint_path
			saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

		# Train results
		for epoch_num, epoch in enumerate(generate_epoch(train_X, train_y, train_seq_lens, 
									FLAGS.num_epochs, FLAGS.batch_size)):
			print "EPOCH:", epoch_num

			sess.run(tf.assign(g['lr'], FLAGS.learning_rate * (FLAGS.learning_rate_decay_factor ** epoch_num)))

			train_loss = []
			train_accuracy = []
			for batch_num, (batch_X, batch_y, batch_seq_lens) in enumerate(epoch):

				batch_seq_lens = batch_seq_lens.reshape((-1, 1))

				_, loss, accuracy = sess.run([g['train_optimizer'], g['loss'], g['accuracy']],
									feed_dict={g['inputs_X']:batch_X, g['targets_y']:batch_y})

				train_loss.append(loss)
				train_accuracy.append(accuracy)

			print
			print "EPOCH %i SUMMARY" % epoch_num
			print "Training loss %.3f" % np.mean(train_loss)
			print "Training accuracy %.3f" % np.mean(train_accuracy)
			print "----------------------"

			# Validation results
			for valid_epoch_num, valid_epoch in enumerate(generate_epoch(valid_X, valid_y, valid_seq_lens, 
													num_epochs=1, batch_size=FLAGS.batch_size)):
				valid_loss = []
				valid_accuracy = []
				for valid_batch_num, (valid_batch_X, valid_batch_y, valid_batch_seq_lens) in enumerate(valid_epoch):
					valid_batch_seq_lens = valid_batch_seq_lens.reshape((-1, 1))

					loss, accuracy = sess.run([g['loss'], g['accuracy']],
									feed_dict={g['inputs_X']:valid_batch_X, g['targets_y']:valid_batch_y})

					valid_loss.append(loss)
					valid_accuracy.append(accuracy)

			print "Validation loss %.3f" % np.mean(valid_loss)
			print "Validation accuracy %.3f" % np.mean(valid_accuracy)
			print "----------------------"

			# Saving the model
			print ("Saving Model at Epoch %i\n" % (epoch_num))
			checkpoint_path = os.path.join(FLAGS.ckpt_dir, 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=epoch_num)





from data_utils import *
from model import *
from train import *

import tensorflow as tf
import os

if __name__ == '__main__':

	X, y = load_data_and_labels()
	vocab_list, vocab_dict, rev_vocab_dict = create_vocabulary(X)
	X, seq_lens = data_to_token_ids(X, vocab_dict)

	# Changes when sampling
	FLAGS.dropout = 0.0
	FLAGS.batch_size = 1

	test_sentence = "It was the worst movie I have ever seen."
	test_sentence = get_tokens(clean_str(test_sentence))
	test_sentence, seq_len = data_to_token_ids([test_sentence], vocab_dict)
	test_sentence = test_sentence[0]
	test_sentence = test_sentence + ([PAD_ID] * (max(len(sentence) for sentence in X) - len(test_sentence)))
	test_sentence = np.array(test_sentence).reshape([1, -1])

	g = model(FLAGS, train=False, sample=True)

	with tf.Session() as sess:
		tf.initialize_all_variables().run()

		# Load old model state if available 
		saver = tf.train.Saver(tf.all_variables())
		ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
		if ckpt and ckpt.model_checkpoint_path:
			print "Loading old model from:", ckpt.model_checkpoint_path
			saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

		probabilities = sess.run([g['probabilities']], feed_dict={g['inputs_X']: test_sentence})

		for index, prob in enumerate(probabilities[0]):
			print rev_vocab_dict[test_sentence[0][index]], prob[1]



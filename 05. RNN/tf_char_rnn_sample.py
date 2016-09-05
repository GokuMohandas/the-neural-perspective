import numpy as np
import tensorflow as tf
import sys
import os
import random
import cPickle

from tf_char_rnn import *

def generate_sample(config, g):


	with tf.Session() as sess:
		tf.initialize_all_variables().run()

		''' Generate sample prediction '''
		state = g['initial_state'].eval()
	
		# Load saved model weights
		saver = tf.train.Saver(tf.all_variables())
		ckpt = tf.train.get_checkpoint_state(config.CKPT_DIR)
		if ckpt and ckpt.model_checkpoint_path:
			print "Loading old model from:", ckpt.model_checkpoint_path
			saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

		# Process state for given SEED_TOKENS
		for char in config.SEED_TOKENS[:-1]:
			word = np.array(config.char_to_idx[char]).reshape(1,1)
			state = sess.run([g['final_state']], feed_dict={g['input_data']:word, g['initial_state']:state})
			state = np.array(state).reshape((1, np.shape(state)[-1]))

		# Sample text for <sample_len> characters
		sample = config.SEED_TOKENS
		prev_char = sample[-1]
		for word_num in range(0, config.SAMPLE_LEN):
			word = np.array(config.char_to_idx[prev_char]).reshape(1,1)
			probs, state = sess.run([g['probabilities'], g['final_state']], feed_dict={g['input_data']:word, g['initial_state']:state})
			state = np.array(state).reshape((1, np.shape(state)[-1]))

			# probs[0] bc probs is 2D array with just one item
			next_char_dist = probs[0]

			# scale the distribution
			next_char_dist /= config.TEMPERATURE # for creativity, higher temperatures produce more nonexistent words BUT more creative samples
			next_char_dist = np.exp(next_char_dist)
			next_char_dist /= sum(next_char_dist)

			if config.SAMPLE_TYPE == 0:
				choice_index = np.argmax(next_char_dist)
			elif config.SAMPLE_TYPE  == 1:
				choice_index = -1
				point = random.random()
				weight = 0.0
				for index in range(0, config.NUM_CLASSES):
					weight += next_char_dist[index]
					if weight >= point:
						choice_index = index
						break
			else:
				raise ValueError("Pick a valid sampling_type!")

			sample += config.idx_to_char[choice_index]
			prev_char = sample[-1]

		print "\nPrediction:"
		print sample

	return sample



if __name__ == '__main__':

	# Load configs
	with open('char_RNN_ckpt_dir/config.pkl', 'rb') as f:
		config = cPickle.load(f)

	# Change config features for sampling
	config.NUM_BATCHES = 1
	config.SEQ_LEN = 1
	config.DROPOUT = 0.0
	g = model(config) 

	generate_sample(config, g)
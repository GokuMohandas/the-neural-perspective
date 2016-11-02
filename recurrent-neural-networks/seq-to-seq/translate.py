from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
import time
import math
import sys

import data_utils
import model

class parameters():

	def __init__(self):

		self.DATA_DIR = "data"
		self.TRAIN_DIR = "train"
		self.MAX_ENGLISH_VOCABULARY_SIZE = 8000
		self.MAX_FRENCH_VOCABULARY_SIZE = 8000

		self.BATCH_SIZE = 50

		self.DROPOUT = 0.0
		self.RNN_UNIT = 'gru' # don't use LSTM without configuring
		self.NUM_HIDDEN_UNITS = 300 # for embedding size and decoding operations
		self.NUM_LAYERS = 2

		self.BUCKETS = data_utils._buckets
		self.LEARNING_RATE = 0.5
		self.LEARNING_RATE_DECAY_FACTOR = 0.99
		self.MAX_GRADIENT_NORM = 5.0

		self.STEPS_PER_CHECKPOINT = 1
		self.NUM_SAMPLES = 5


def get_data(FLAGS):

	# Load train and test sets
	train_set, test_set = data_utils.load_data(FLAGS, train_ratio=0.8)
	for bucket_id, (_, _) in enumerate(FLAGS.BUCKETS):
		print ("BUCKET: %i, TRAIN: %d, TEST: %d" % (bucket_id,
			len(train_set[bucket_id]), len(test_set[bucket_id])))

	return train_set, test_set

def create_model(session, forward_only, FLAGS):

	if forward_only:
		dropout = 0.0
	else:
		dropout=FLAGS.DROPOUT

	s2smodel = model.s2s_model(
		source_vocab_size=FLAGS.MAX_ENGLISH_VOCABULARY_SIZE,
		target_vocab_size=FLAGS.MAX_FRENCH_VOCABULARY_SIZE,
		buckets=FLAGS.BUCKETS,
		size=FLAGS.NUM_HIDDEN_UNITS,
		num_layers=FLAGS.NUM_LAYERS,
		max_gradient_norm=FLAGS.MAX_GRADIENT_NORM,
		dropout=dropout,
		batch_size=FLAGS.BATCH_SIZE,
		learning_rate=FLAGS.LEARNING_RATE,
		learning_rate_decay_factor=FLAGS.LEARNING_RATE_DECAY_FACTOR,
		forward_only=forward_only)

	ckpt = tf.train.get_checkpoint_state(FLAGS.TRAIN_DIR)
	if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
		print("Restoring old model parameters from %s" % ckpt.model_checkpoint_path)
		s2smodel.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created new model.")
		session.run(tf.initialize_all_variables())

	return s2smodel

def train(FLAGS):

	with tf.Session() as sess:

		print ("Making the model")
		model = create_model(sess, forward_only=False, FLAGS=FLAGS)
		print ("Model made!")

		# Get train and test datasets
		train_set, test_set = get_data(FLAGS)

		# Method to choose bucket_id
		train_bucket_sizes = [len(train_set[b]) for b in xrange(len(FLAGS.BUCKETS))]
		train_total_size = float(sum(train_bucket_sizes))
		train_buckets_scale = [sum(train_bucket_sizes[:i+1]) / train_total_size
			for i in xrange(len(train_bucket_sizes))]

		# Training begins
		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []
		while True:
			random_number_01 = np.random.random_sample()
			bucket_id = min([i for i in xrange(len(train_buckets_scale))
				if train_buckets_scale[i] > random_number_01])

			# Retrieve a batch and do one training step
			start_time = time.time()
			encoder_inputs, decoder_inputs, target_weights = model.get_batch(
				train_set, bucket_id)

			print ("ENCODER INPUTS", np.shape(encoder_inputs))
			print ("DECODER INPUTS", np.shape(decoder_inputs))
			print ("TARGET WEIGHTS", np.shape(target_weights))

			_, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
				target_weights, bucket_id, forward_only=False)
			step_time += (time.time() - start_time) / FLAGS.STEPS_PER_CHECKPOINT
			loss += step_loss / FLAGS.STEPS_PER_CHECKPOINT
			current_step += 1

			print (step_loss)

			# Show inference results every FLAGS.STEPS_PER_CHECKPOINT
			if current_step % FLAGS.STEPS_PER_CHECKPOINT == 0:

				# Print statistics for the previous epoch.
				perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
				print ("global step %d learning rate %.4f step-time %.2f perplexity "
					"%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
						step_time, perplexity))

				# Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)

				# Save checkpoint and zero timer and loss.
				if not os.path.isdir(FLAGS.TRAIN_DIR):
					os.makedirs(FLAGS.TRAIN_DIR)
				checkpoint_path = os.path.join(FLAGS.TRAIN_DIR, "translate.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				step_time, loss = 0.0, 0.0

				# Run evals on development set and print their perplexity.
				for bucket_id in xrange(len(FLAGS.BUCKETS)):
					encoder_inputs, decoder_inputs, target_weights = model.get_batch(
						test_set, bucket_id)
					_, eval_loss, _ = model.step(sess, encoder_inputs,
						decoder_inputs, target_weights, bucket_id, True)
					eval_ppx = math.exp(float(eval_loss)) \
						if eval_loss < 300 else float("inf")
					print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
				sys.stdout.flush()

				# Generate some samples
				os.system('python translate.py decode')

def decode(FLAGS):

	with tf.Session() as sess:
		# Create model and load parameters.
		SAMPLING_FLAGS = parameters()
		SAMPLING_FLAGS.BATCH_SIZE = 1
		model = create_model(sess, forward_only=True, FLAGS=SAMPLING_FLAGS)

		# Load vocabularies.
		en_vocab_path = os.path.join(FLAGS.DATA_DIR,
			"vocab%d.en" % FLAGS.MAX_ENGLISH_VOCABULARY_SIZE)
		fr_vocab_path = os.path.join(FLAGS.DATA_DIR,
			"vocab%d.fr" % FLAGS.MAX_FRENCH_VOCABULARY_SIZE)
		en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
		_, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

		# Get train and test datasets
		train_set, test_set = get_data(FLAGS)

		# Method to choose bucket_id
		train_bucket_sizes = [len(train_set[b]) for b in xrange(len(FLAGS.BUCKETS))]
		train_total_size = float(sum(train_bucket_sizes))
		train_buckets_scale = [sum(train_bucket_sizes[:i+1]) / train_total_size
			for i in xrange(len(train_bucket_sizes))]

		# Training begins
		for sample in xrange(FLAGS.NUM_SAMPLES):
			random_number_01 = np.random.random_sample()
			bucket_id = min([i for i in xrange(len(train_buckets_scale))
				if train_buckets_scale[i] > random_number_01])

			encoder_inputs, decoder_inputs, target_weights = model.get_batch(
				train_set, bucket_id)

			_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
				target_weights, bucket_id, forward_only=True)
			# This is a greedy decoder - outputs are just argmaxes of output_logits.
			outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
			# If there is an EOS symbol in outputs, cut them at that point.
			if data_utils.EOS_ID in outputs:
				outputs = outputs[:outputs.index(data_utils.EOS_ID)]
			# Print out French sentence corresponding to outputs.
			print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) \
				for output in outputs]))

def translate_sentence(FLAGS):

	""" Translate a sentence from terminal input """

	with tf.Session() as sess:
		# Create model and load parameters.
		SAMPLING_FLAGS = parameters()
		SAMPLING_FLAGS.BATCH_SIZE = 1
		model = create_model(sess, forward_only=True, FLAGS=SAMPLING_FLAGS)

		# Load vocabularies.
		en_vocab_path = os.path.join(FLAGS.DATA_DIR,
			"vocab%d.en" % FLAGS.MAX_ENGLISH_VOCABULARY_SIZE)
		fr_vocab_path = os.path.join(FLAGS.DATA_DIR,
			"vocab%d.fr" % FLAGS.MAX_FRENCH_VOCABULARY_SIZE)
		en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
		_, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

		# Decode from standard input.
		sys.stdout.write("> ")
		sys.stdout.flush()
		sentence = sys.stdin.readline()
		while sentence:
			# Get token-ids for the input sentence.
			token_ids = data_utils.sentence_to_token_ids(
				tf.compat.as_bytes(sentence), en_vocab)
			# Which bucket does it belong to?
			bucket_id = min([b for b in xrange(len(FLAGS.BUCKETS))
				if FLAGS.BUCKETS[b][0] > len(token_ids)])
			# Get a 1-element batch to feed the sentence to the model.
			encoder_inputs, decoder_inputs, target_weights = model.get_batch(
				{bucket_id: [(token_ids, [])]}, bucket_id)
			# Get output logits for the sentence.
			_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
				target_weights, bucket_id, True)
			# This is a greedy decoder - outputs are just argmaxes of output_logits.
			outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
			# If there is an EOS symbol in outputs, cut them at that point.
			if data_utils.EOS_ID in outputs:
				outputs = outputs[:outputs.index(data_utils.EOS_ID)]
			# Print out French sentence corresponding to outputs.
			print(" ".join([tf.compat.as_str(
				rev_fr_vocab[output]) for output in outputs]))
			print("> ",end="")
			sys.stdout.flush()
			sentence = sys.stdin.readline()

if __name__ == '__main__':

	# configs
	FLAGS = parameters()

	if sys.argv[1] == 'train':
		train(FLAGS)
	elif sys.argv[1] == 'decode':
		decode(FLAGS)
	elif sys.argv[1] == 'translate':
		translate_sentence(FLAGS)







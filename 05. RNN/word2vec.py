import tensorflow as tf
import zipfile
import re
import collections
import numpy as np
import random

class parameters():

	def __init__(self):
		self.FILENAME = 'data/hp.txt'
		self.NUM_EPOCHS = 1000

		self.VOCAB_SIZE = 2000
		self.SEQ_LEN = 128 # number of words in input
		self.NUM_SKIPS = 2 # skip the first <num_skips/2> words at the beginning and same at end
		self.SKIP_WINDOW = 1 # get the <skip_window> context words to the right and left of target
		self.EMBEDDING_SIZE = 128
		self.NUM_SAMPLED = 64 # number of random words for negative sampling

		self.VALID_LEN = 16 # number of words to get neighbors for
		self.VALID_WINDOW = 100 # words to sample from 
		self.TOP_K = 5 # show top 5 similar words
		self.SAMPLE_EVERY = 100 # show neighbors every <SAMPLE_EVERY> epochs


def get_data(config):
	with open(config.FILENAME, 'r') as f:
		text = f.read()
	# Get only the words without punctuation	
	words = re.compile('\w+').findall(text)
	# lowercase all the words
	words = [word.lower() for word in words]
	return words

def build_dataset(config, words):

	# Get counts for top <VOCAB_SIZE> words
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(config.VOCAB_SIZE - 1))

	common_words = []
	for word, freq in count:
		common_words.append(word)

	# Dicts
	word_to_idx = {char:i for i,char in enumerate(common_words)}
	idx_to_word = {i:char for i,char in enumerate(common_words)}

	# Counts for unknown
	unknown_count = 0
	for word in words:
		if word not in word_to_idx:
			unknown_count +=1
	count[0] = ('UNK', unknown_count)

	# Data
	data = [word_to_idx[word] if word in common_words else word_to_idx['UNK'] for word in words]

	#print "Top 5 words:", count[:5]
	#print "Sample:", words[:10]
	#print "Data:", data[:10]

	return data, count, word_to_idx, idx_to_word

def generate_batch(config, data):

	# get number of batches
	config.NUM_BATCHES = len(data) // config.SEQ_LEN
	data = data[:config.NUM_BATCHES * config.SEQ_LEN]

	# Create batches
	for batch_num in range(config.NUM_BATCHES):
		input_data = data[batch_num*config.SEQ_LEN:(batch_num+1)*config.SEQ_LEN]
		X, y = skip_gram(input_data, config.NUM_SKIPS, config.SKIP_WINDOW)
		yield X, y

def generate_epochs(config, data):
	for epoch_num in range(config.NUM_EPOCHS):
		yield generate_batch(config, data)

def skip_gram(input_data, num_skips, skip_window):

	# skips
	data = input_data[num_skips//2:]
	data = data[:-num_skips//2]

	X, y = [], []
	for word_num, word in enumerate(data):
		word_num += num_skips//2 # account for skips
		for window in range(1, skip_window+1):
			X.append(word)
			X.append(word)
			y.append(input_data[word_num-window]) # use input_data bc data doesnt have skipped words
			y.append(input_data[word_num+window])
	
	X = np.array(X)
	y = np.array(y).reshape((-1, 1))
	return X, y

def model(config):

	# Get random array of <VALID_LEN> indexes out of top <VALID_WINDOW>
	valid_examples = np.array(random.sample(range(config.VALID_WINDOW),
						config.VALID_LEN))

	# Placeholders
	train_X = tf.placeholder(tf.int32, shape=[(config.SEQ_LEN-config.NUM_SKIPS)*2])
	train_y = tf.placeholder(tf.int32, shape=[(config.SEQ_LEN-config.NUM_SKIPS)*2, 1])
	valid_X = tf.constant(valid_examples, dtype=tf.int32)

	# Weights
	W_input = tf.Variable(
			   tf.random_uniform([config.VOCAB_SIZE, 
			   config.EMBEDDING_SIZE], -1.0, 1.0)) # embedding weights
	W_softmax = tf.Variable(
			  tf.truncated_normal([config.VOCAB_SIZE,
			  config.EMBEDDING_SIZE]))
	b_softmax = tf.Variable(tf.zeros([config.VOCAB_SIZE]))

	# Train
	embeddings = tf.nn.embedding_lookup(W_input, train_X)
	loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(W_softmax, 
					  b_softmax, embeddings, train_y, 
					  config.NUM_SAMPLED, config.VOCAB_SIZE))
	optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

	# Cosine similarity with valid_X to get similar terms for each word
	norm = tf.sqrt(tf.reduce_sum(tf.square(W_input), 1, keep_dims=True))
	normalized_embeddings = W_input / norm # denom of cosine similarity
	valid_embeddings = tf.nn.embedding_lookup(
					normalized_embeddings, valid_X)
	similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

	return dict(train_X=train_X, train_y=train_y, valid_examples=valid_examples,
			  loss=loss, optimizer=optimizer, similarity=similarity)

def train(config, data, g, idx_to_word):

	# Start tf session
	with tf.Session() as sess:
		tf.initialize_all_variables().run()

		for epoch_num, epoch in enumerate(generate_epochs(config, data)):
			for batch_num, (X, y) in enumerate(epoch):
				feed_dict = {g['train_X']: X, g['train_y']: y}
				loss, _ = sess.run([g['loss'], g['optimizer']],
								feed_dict=feed_dict)
			
			print "\nEPOCH: %i, LOSS: %.3f" % (epoch_num, loss)
			
			# Similarity results for few terms
			if epoch_num%config.SAMPLE_EVERY == 0:
				similarity = g['similarity'].eval()
				valid_examples = g['valid_examples']

				for word_num in range(config.VALID_LEN):
					word = idx_to_word[valid_examples[word_num]]
					nearest_words = (-similarity[word_num, :]).argsort()[1:config.TOP_K+1] # 1 bc most similar will be word itself
					print "Close to", word, ":", [idx_to_word[word_idx] for word_idx in nearest_words[:config.TOP_K]]



if __name__ == '__main__':
	config = parameters()
	words = get_data(config)
	data, count, word_to_idx, idx_to_word = build_dataset(config, words)

	g = model(config)
	train(config, data, g, idx_to_word)










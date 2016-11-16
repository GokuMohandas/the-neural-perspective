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
        self.VALID_WINDOW = 100 # words to sample from for sampled softmax
        self.TOP_K = 5 # show top 5 similar words
        self.SAMPLE_EVERY = 100 # show neighbors every <SAMPLE_EVERY> epochs


def get_data(FLAGS):
    with open(FLAGS.FILENAME, 'r') as f:
        text = f.read()
    # Get only the words without punctuation
    words = re.compile('\w+').findall(text)
    # lowercase all the words
    words = [word.lower() for word in words]
    return words

def build_dataset(FLAGS, words):

    # Get counts for top <VOCAB_SIZE> words
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(FLAGS.VOCAB_SIZE - 1))

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
    data = [word_to_idx[word]
        if word in common_words else word_to_idx['UNK'] for word in words]

    #print "Top 5 words:", count[:5]
    #print "Sample:", words[:10]
    #print "Data:", data[:10]

    return data, count, word_to_idx, idx_to_word

def generate_batch(FLAGS, data):

    # get number of batches
    FLAGS.NUM_BATCHES = len(data) // FLAGS.SEQ_LEN
    data = data[:FLAGS.NUM_BATCHES * FLAGS.SEQ_LEN]

    # Create batches
    for batch_num in range(FLAGS.NUM_BATCHES):
        input_data = data[batch_num*FLAGS.SEQ_LEN:(batch_num+1)*FLAGS.SEQ_LEN]
        X, y = skip_gram(input_data, FLAGS.NUM_SKIPS, FLAGS.SKIP_WINDOW)
        yield X, y

def generate_epochs(FLAGS, data):
    for epoch_num in range(FLAGS.NUM_EPOCHS):
        yield generate_batch(FLAGS, data)

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
            # use input_data bc data doesnt have skipped words
            y.append(input_data[word_num-window])
            y.append(input_data[word_num+window])

    X = np.array(X)
    y = np.array(y).reshape((-1, 1))
    return X, y

class model(object):

    def __init__(self, FLAGS):

        # Get random array of <VALID_LEN> indexes out of top <VALID_WINDOW>
        self.valid_examples = np.array(random.sample(range(FLAGS.VALID_WINDOW),
                            FLAGS.VALID_LEN))

        # Placeholders
        self.train_X = tf.placeholder(tf.int32,
            shape=[(FLAGS.SEQ_LEN-FLAGS.NUM_SKIPS)*2])
        self.train_y = tf.placeholder(tf.int32,
            shape=[(FLAGS.SEQ_LEN-FLAGS.NUM_SKIPS)*2, 1])
        self.valid_X = tf.constant(self.valid_examples, dtype=tf.int32)

        # Weights
        W_input = tf.Variable(
                   tf.random_uniform([FLAGS.VOCAB_SIZE,
                   FLAGS.EMBEDDING_SIZE], -1.0, 1.0)) # embedding weights
        W_softmax = tf.Variable(
                  tf.truncated_normal([FLAGS.VOCAB_SIZE,
                  FLAGS.EMBEDDING_SIZE]))
        b_softmax = tf.Variable(tf.zeros([FLAGS.VOCAB_SIZE]))

        # Train
        self.embeddings = tf.nn.embedding_lookup(W_input, self.train_X)
        self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(W_softmax,
                          b_softmax, self.embeddings, self.train_y,
                          FLAGS.NUM_SAMPLED, FLAGS.VOCAB_SIZE))
        self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)

        # Cosine similarity with valid_X to get similar terms for each word
        self.norm = tf.sqrt(tf.reduce_sum(tf.square(W_input), 1, keep_dims=True))
        self.normalized_embeddings = W_input / self.norm # denom of cosine similarity
        self.valid_embeddings = tf.nn.embedding_lookup(
                        self.normalized_embeddings, self.valid_X)
        self.similarity = tf.matmul(self.valid_embeddings,
            tf.transpose(self.normalized_embeddings))

    def step(self, sess, batch_X, batch_y):

        input_feed = {self.train_X: batch_X, self.train_y: batch_y}
        output_feed = [self.loss, self.optimizer]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]

def create_model(sess, FLAGS):
    sg_model = model(FLAGS)
    print("Created new model.")
    sess.run(tf.initialize_all_variables())
    return sg_model

def train(FLAGS, data, idx_to_word):

    # Start tf session
    with tf.Session() as sess:

        # Load the model
        model = create_model(sess, FLAGS)

        for epoch_num, epoch in enumerate(generate_epochs(FLAGS, data)):
            for batch_num, (X, y) in enumerate(epoch):
                loss, _ = model.step(sess, X, y)

            print "\nEPOCH: %i, LOSS: %.3f" % (epoch_num, loss)

            # Similarity results for few terms
            if epoch_num%FLAGS.SAMPLE_EVERY == 0:
                similarity = model.similarity.eval()
                valid_examples = model.valid_examples

                for word_num in range(FLAGS.VALID_LEN):
                    word = idx_to_word[valid_examples[word_num]]
                    nearest_words = (-similarity[word_num, :]).argsort()[1:FLAGS.TOP_K+1] # 1 bc most similar will be word itself
                    print "Close to", word, ":", [idx_to_word[word_idx] for word_idx in nearest_words[:FLAGS.TOP_K]]



if __name__ == '__main__':
    FLAGS = parameters()
    words = get_data(FLAGS)
    data, count, word_to_idx, idx_to_word = build_dataset(FLAGS, words)
    train(FLAGS, data, idx_to_word)










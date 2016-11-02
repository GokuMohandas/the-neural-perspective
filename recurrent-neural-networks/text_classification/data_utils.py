import numpy as np
import re
import itertools
from collections import Counter

# sentence polarity dataset v1.0 from http://www.cs.cornell.edu/people/pabo/movie-review-data/

# Processing tokens
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def get_tokens(sent):

    words = []
    for space_separated_fragment in sent.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels():

    # Load the data
    positive_examples = list(open("data/rt-polaritydata/rt-polarity.pos", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    positive_examples = [get_tokens(clean_str(sent)) for sent in positive_examples]
    negative_examples = list(open("data/rt-polaritydata/rt-polarity.neg", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    negative_examples = [get_tokens(clean_str(sent)) for sent in negative_examples]
    X = positive_examples + negative_examples

    # Labels
    positive_labels = [[0,1] for _ in positive_examples]
    negative_labels = [[1,0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    print "Total: %i, NEG: %i, POS: %i" % (len(y), np.sum(y[:, 0]), np.sum(y[:, 1]))

    return X, y

def create_vocabulary(X, max_vocabulary_size=5000):

    vocab = {}
    for sentence in X:
        for word in sentence:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1

    # Get list of all vocab words starting with [_PAD, _GO, _EOS, _UNK]
    # and then words sorted by count
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    vocab_list = vocab_list[:max_vocabulary_size]

    vocab_dict = dict((x,y) for (y,x) in enumerate(vocab_list))
    rev_vocab_dict = {v: k for k, v in vocab_dict.items()}

    print "Total of %i unique tokens" % len(vocab_list)
    return vocab_list, vocab_dict, rev_vocab_dict

def sentence_to_token_ids(sentence, vocab_dict):

    # get value for w if it is in vocab dict else return UNK_ID = 3
    return [vocab_dict.get(word, UNK_ID) for word in sentence]

def data_to_token_ids(X, vocab_dict):

    max_len = max(len(sentence) for sentence in X)
    seq_lens = []

    data_as_tokens = []
    for line in X:
        token_ids = sentence_to_token_ids(line, vocab_dict)
        # Padding
        data_as_tokens.append(token_ids + [PAD_ID]*(max_len - len(token_ids)))
        # Maintain original seq lengths for dynamic RNN
        seq_lens.append(len(token_ids))

    return data_as_tokens, seq_lens

def split_data(X, y, seq_lens, train_ratio=0.8):

    X = np.array(X)
    seq_lens = np.array(seq_lens)
    data_size = len(X)

    # Shuffle the data
    shuffle_indices = np.random.permutation(np.arange(data_size))
    X, y, seq_lens = X[shuffle_indices], y[shuffle_indices], \
                     seq_lens[shuffle_indices]

    # Split into train and validation set
    train_end_index = int(train_ratio*data_size)
    train_X = X[:train_end_index]
    train_y = y[:train_end_index]
    train_seq_lens = seq_lens[:train_end_index]
    valid_X = X[train_end_index:]
    valid_y = y[train_end_index:]
    valid_seq_lens = seq_lens[train_end_index:]

    return train_X, train_y, train_seq_lens, valid_X, valid_y, valid_seq_lens

def generate_epoch(X, y, seq_lens, num_epochs, batch_size):

    for epoch_num in range(num_epochs):
        yield generate_batch(X, y, seq_lens, batch_size)

def generate_batch(X, y, seq_lens, batch_size):

    data_size = len(X)

    num_batches = (data_size // batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield X[start_index:end_index], y[start_index:end_index], \
              seq_lens[start_index:end_index]

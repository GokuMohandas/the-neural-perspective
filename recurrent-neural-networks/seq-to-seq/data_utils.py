# Modified from https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
from six.moves import urllib
from tensorflow.python.platform import gfile
import random
import numpy as np

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(BR"\d")

#[(source sentence length, target sentence length), (), ...]
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

train_url = "http://www.statmt.org/wmt10/training-giga-fren.tar"
test_url = "http://www.statmt.org/wmt15/dev-v2.tgz"

def download_data(directory, filename, url):
	""" download data from url into directory/filename """
	if not os.path.exists(directory):
		os.mkdir(directory)
	
	filepath = os.path.join(directory, filename)
	if not os.path.exists(filepath):
		filepath, _ = urllib.request.urlretrieve(url, filepath)

	return filepath

def get_dev_set(directory):
	""" Download the WMT en-fr dev corpus """
	dev_name = "newstest2013"
	dev_path = os.path.join(directory, dev_name)
	if not (gfile.Exists(dev_path + ".fr") and gfile.Exists(dev_path + ".en")):
		dev_file = download_data(directory, "dev-v2.tgz", test_url)
		with tarfile.open(dev_file, "r:gz") as dev_tar:
			fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
			en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
			
			# take out the dev/ part of the file name
			fr_dev_file.name = dev_name + ".fr"
			en_dev_file.name = dev_name + ".en"

			dev_tar.extract(fr_dev_file, directory)
			dev_tar.extract(en_dev_file, directory)
	return dev_path

def basic_tokenizer(sentence):
	""" Split sentence into list of tokens """
	words = []
	for space_separated_item in sentence.strip().split():
		words.extend(_WORD_SPLIT.split(space_separated_item))
	return [w for w in words if w] # if w removes the ""

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
				 normalize_digits=True):
	""" We will create a file with all the vocabulary words up to
	    max_vocabulary_size. 
	
	Args:
		vocabulary_path: path where the vocabulary will be created.
		data_path: data file that will be used to create vocabulary.
		max_vocabulary_size: limit on the size of the created vocabulary.
		normalize_digits: Boolean; if true, all digits are replaced by 0s.

	"""
	if not gfile.Exists(vocabulary_path):
		vocab = {}
		with gfile.GFile(data_path, "rb") as f:
			counter = 0
			for line in f:
				counter += 1
				if counter % 1000 == 0:
					print (" processing line %d" % counter)
				tokens = basic_tokenizer(line)
				for w in tokens:
					word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
					if word in vocab:
						vocab[word] += 1
					else:
						vocab[word] = 1
				vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, 
											reverse=True)
				if len(vocab_list) > max_vocabulary_size:
					vocab_list = vocab_list[:max_vocabulary_size]
				with gfile.GFile(vocabulary_path, "wb") as vocab_file:
					for word in vocab_list:
						vocab_file.write(word + b"\n")

def initialize_vocabulary(vocabulary_path):
	""" We will create a dict to map tokens to integers and a 
		reverse list which will reverse the vocabulary mapping. 

	Args:
		vocabulary_path: path where the vocabulary will be created.

	"""
	if gfile.Exists(vocabulary_path):
		vocab_list = []
		with gfile.GFile(vocabulary_path, "rb") as f:
			vocab_list.extend(f.readlines())
		vocab_list = [line.strip() for line in vocab_list]

		vocab = dict([(x, y) for (y, x) in enumerate(vocab_list)])
		return vocab, vocab_list
	else:
		raise ValueError("File Not Found: %s" % vocabulary_path)

def sentence_to_token_ids(sentence, vocab_dict, normalize_digits=True):
	""" Convert string sentence to token ids.

	Args:
		sentence: sentence to convert to token ids
		vocabulary: dict mapping tokens to integers
		normalize_digits: boolean; if true, the digits will be replaced by "0"
	"""

	words = basic_tokenizer(sentence)
	if not normalize_digits:
		# replace words not in vocab_dict with UNK_ID
		return [vocab_dict.get(w, UNK_ID) for w in words]
	else:
		return [vocab_dict.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path,
				  normalize_digits=True):
	""" Tokensize the data into token ids used the vocab file.

	Args:
		data_path: source to data file with one sentence per line
		target_path: path where sentences of token ids will be stored
		vocabulary_path: path to the vocabulary file
		normalize_digits: boolean; if true, the digits will be replaced by "0"

	"""

	if not gfile.Exists(target_path):
		vocab_dict, vocab_list = initialize_vocabulary(vocabulary_path)
		with gfile.GFile(data_path, "rb") as data_file:
			with gfile.GFile(target_path, "w") as tokens_file:
				counter = 0
				for line in data_file:
					counter += 1
					if counter % 1000 == 0:
						print (" tokenizing line %d" % counter)
					token_ids = sentence_to_token_ids(line, vocab_dict)
					tokens_file.write(" ".join([str(tok) for tok in 
						token_ids]) + "\n")

def read_data(source_path, target_path, max_size=None, verbose=False):
	""" Read the data from source (en) and target (fr) files and
		place into buckets. Not each line in source corresponds to
		that same line number in target file.

	Args:
		source_path: path to the file with sentences as token ids for source language.
		target_path: path to the file with sentences as token ids for target language.
		max_size: max number of lines to read. If None, we read all.
	
	"""

	data_set = [[] for _ in _buckets]
	with gfile.GFile(source_path, "r") as source_file:
		with gfile.GFile(target_path, "r") as target_file:
			source, target = source_file.readline(), target_file.readline()
			counter = 0
			while source and target and (not max_size or counter < max_size):
				counter += 1

				if verbose:
					if counter % 1000 == 0:
						print (" reading data line %d" % counter)

				source_ids = [int(x) for x in source.split()]
				target_ids = [int(x) for x in target.split()]
				target_ids.append(EOS_ID)

				for bucket_id, (source_size, target_size) in enumerate(_buckets):
					if (len(source_ids) < source_size and len(target_ids) < target_size):
						
						data_set[bucket_id].append([source_ids, target_ids])
						break

				source, target = source_file.readline(), target_file.readline()
	return data_set

def load_data(FLAGS, train_ratio=0.8):

	""" Uses all the functions in data_utils to download the data,
	clean it, tokenize it, split into buckets, and finally here, we 
	split into train and test sets. 

	NOTE: We only download the test (dev) set and then from that we create
	our train and test set in order to save memory on my machine. To
	actually train the model well, download the train set by following
	the instructions on the tensorflow documentation for seq-to-seq models.

	"""

	# Load the data into .en and .fr files
	dev_path = get_dev_set("data")

	# Get the english and french vocabularies
	en_vocab_path = os.path.join(FLAGS.DATA_DIR,
				 "vocab%d.en" % FLAGS.MAX_ENGLISH_VOCABULARY_SIZE)
	fr_vocab_path = os.path.join(FLAGS.DATA_DIR,
				 "vocab%d.fr" % FLAGS.MAX_FRENCH_VOCABULARY_SIZE)
	create_vocabulary(en_vocab_path, dev_path + ".en",
						    FLAGS.MAX_ENGLISH_VOCABULARY_SIZE)
	create_vocabulary(fr_vocab_path, dev_path + ".fr",
						    FLAGS.MAX_FRENCH_VOCABULARY_SIZE)
	
	# Convert data to tokens
	en_dev_ids_path = dev_path + (".ids%d.en" % 
							FLAGS.MAX_ENGLISH_VOCABULARY_SIZE)
	fr_dev_ids_path = dev_path + (".ids%d.fr" % 
							FLAGS.MAX_FRENCH_VOCABULARY_SIZE)
	data_to_token_ids(dev_path+".en", en_dev_ids_path, en_vocab_path)
	data_to_token_ids(dev_path+".fr", fr_dev_ids_path, fr_vocab_path)

	# Create dicts to go from word->index and index->word
	en_vocab_dict, en_vocab_list = initialize_vocabulary(en_vocab_path)
	fr_vocab_dict, fr_vocab_list = initialize_vocabulary(fr_vocab_path)
	rev_en_vocab_dict = dict([x, y] for (x, y) in enumerate(en_vocab_list))
	rev_fr_vocab_dict = dict([x, y] for (x, y) in enumerate(fr_vocab_list))

	# Place tokenized data into buckets
	data_set = read_data(en_dev_ids_path, fr_dev_ids_path, verbose=False)

	# Sample
	#print ["".join(rev_en_vocab_dict[token_id]) for token_id in  data_set[0][0][0]]
	#print ["".join(rev_fr_vocab_dict[token_id]) for token_id in  data_set[0][0][1]]

	# Split data into train and test
	train_set = [[] for _ in FLAGS.BUCKETS]
	test_set = [[] for _ in FLAGS.BUCKETS]
	for bucket_id, (_, _) in enumerate(FLAGS.BUCKETS):
		train_last_index = int(len(data_set[bucket_id])*0.8)
		train_set[bucket_id] = data_set[bucket_id][:train_last_index]
		test_set[bucket_id] = data_set[bucket_id][train_last_index:]
	
	return train_set, test_set







# encoding=utf8

import sys
import cPickle
import nltk
nltk.download('punkt')

# For special characters
reload(sys)
sys.setdefaultencoding('utf8')

def make_sentences(datafile):
    """
    We will break the raw text in datafile into
    sentences. The nltk package will handle edge cases
    (ex. Mr. Potter) and will give us the list of sentences.
    """

    with open(datafile, 'rb') as f:
        text = f.read()

    sentences = nltk.tokenize.sent_tokenize(text)
    return sentences

def prepare_data():

    english_data = 'data/my_en.txt'
    spanish_data = 'data/my_sp.txt'

    # Break into sentences
    english_sentences = make_sentences(english_data)
    spanish_sentences = make_sentences(spanish_data)
    print "We have %i english sentences and %i spanish sentences." % (
        len(english_sentences), len(spanish_sentences))

    # Store sentences into data file
    with open('data/en.p', 'wb') as f:
        cPickle.dump(english_sentences, f)
    with open('data/sp.p', 'wb') as f:
        cPickle.dump(spanish_sentences, f)


if __name__ == '__main__':
    prepare_data()
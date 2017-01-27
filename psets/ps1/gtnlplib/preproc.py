import nltk
import pandas as pd
from collections import Counter

def tokenize_and_downcase(string,vocab=None):
    """for a given string, corresponding to a document:
    - tokenize first by sentences and then by word
    - downcase each token
    - return a Counter of tokens and frequencies.

    :param string: input document
    :returns: counter of tokens and frequencies
    :rtype: Counter

    """
    bow = Counter()
    sents = nltk.sent_tokenize(string)
    for sent in sents:
        for token in nltk.word_tokenize(sent):
            bow[token.lower()] += 1
    return bow


### Helper code

def read_data(csvfile,labelname,preprocessor=lambda x : x):
    # note that use of utf-8 encoding to read the file
    df = pd.read_csv(csvfile,encoding='utf-8')
    return df[labelname].values,[preprocessor(string) for string in df['text'].values]

def get_corpus_counts(list_of_bags_of_words):
    counts = Counter()
    for bow in list_of_bags_of_words:
        for key,val in bow.iteritems():
            counts[key] += val
    return counts

### Secret bakeoff code
def custom_preproc(string):
    """for a given string, corresponding to a document, tokenize first by sentences and then by word; downcase each token; return a Counter of tokens and frequencies.

    :param string: input document
    :returns: counter of tokens and frequencies
    :rtype: Counter

    """
    # benchmark result: 0.78
	# prev: 0.762
    stemmer = nltk.stem.PorterStemmer()
    feats = Counter()
    sents = nltk.sent_tokenize(string)
    # TODO sentence count feature
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        for token in tokens:
            tok = token.lower()
            feats[tok] += 1 # regular features
            feats["STEM_" + stemmer.stem(tok)] += 1 # stem features
			feats["LEN_" + str(len(token))] += 1 # word length feature
            four = min(len(tok),4)
            for i in xrange(1, four):
                feats["PRE_" + tok[:i]] += 1 # prefix feature
                feats["SUF_" + tok[-i:]] += 1 # suffix feature
            if tok != token:
                feats["CAPITALIZED"] += 1 # word shape feature
        for i in xrange(len(tokens) - 1):
            feats[tokens[i].lower() + "_" + tokens[i+1].lower()] += 1 # bigram features
    return feats

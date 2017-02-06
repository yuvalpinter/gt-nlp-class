import numpy as np #hint: np.log
import sys
from collections import defaultdict,Counter
from gtnlplib import scorer, most_common,preproc
from gtnlplib.constants import OFFSET

def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    counts = defaultdict(int)
    for x_i,y_i in zip(x,y):
        if y_i == label:
            for w,c in x_i.iteritems():
                counts[w] += c
    return counts
    
def estimate_pxy(x,y,label,smoothing,vocab):
    """Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    """
    counts = defaultdict(float)
    for w in vocab:
        counts[w] = smoothing
    for l_w,ct in get_corpus_counts(x,y,label).iteritems():
        counts[l_w] += ct
    total_count = sum(counts.values())
    if total_count <= 0:
        return defaultdict(float)
    log_probs = defaultdict(float, {w:np.log(float(c)/total_count) for w,c in counts.iteritems() if c > 0.0})
    return log_probs

def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    labels = set(y) 
    doc_counts = defaultdict(float)
    
    for y_i in y:
        doc_counts[y_i] += 1
    
    vocab = set.union(*[set(x_i.keys()) for x_i in x])
    corpus_size = sum(doc_counts.values())
    
    weights = defaultdict(float)
    for l in labels:
        log_prior_l = np.log(doc_counts[l]) - np.log(corpus_size)
        weights[(l, OFFSET)] = log_prior_l
        for w,log_prob in estimate_pxy(x,y,l,smoothing,vocab).iteritems():
            weights[(l, w)] = log_prob

    return weights

def estimate_nb_tagger(counters,smoothing):
    """build a tagger based on the naive bayes classifier, which correctly accounts for the prior P(Y)

    :param counters: dict of word-tag counters, from most_common.get_tag_word_counts
    :param smoothing: value for lidstone smoothing
    :returns: classifier weights
    :rtype: defaultdict

    """
    sorted_tags = sorted(counters.keys())
    weights = estimate_nb([counters[t] for t in sorted_tags], sorted_tags, 0.01)
    total_sum = 0
    for c in counters.values():
        total_sum += sum(c.values())
    log_sum = np.log(total_sum)
    for t,c in counters.iteritems():
        weights[(t, OFFSET)] = np.log(sum(c.values())) - log_sum
    return weights

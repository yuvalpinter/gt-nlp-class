from gtnlplib.preproc import get_corpus_counts
from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation

import numpy as np
from collections import defaultdict

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
    
    weights = {}
    for l in labels:
        log_prior_l = np.log(doc_counts[l]) - np.log(corpus_size)
        weights[(l, OFFSET)] = log_prior_l
        for w,log_prob in estimate_pxy(x,y,l,smoothing,vocab).iteritems():
            weights[(l, w)] = log_prob

    return weights
   
argmin = lambda x : min(x.iteritems(),key=lambda y : y[1])[0]

def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    """find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values to try
    :returns: best smoothing value, scores of all smoothing values
    :rtype: float, dict

    """
    scores = {}
    labels = set(y_tr)
    for alpha in smoothers:
        theta = estimate_nb(x_tr,y_tr,alpha)
        y_hat = clf_base.predict_all(x_dv,theta,labels)
        scores[alpha] = evaluation.acc(y_hat,y_dv)
    return argmin(scores), scores

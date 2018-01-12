from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import START_TAG, TRANS, END_TAG, EMIT, OFFSET
from gtnlplib import naive_bayes, most_common
import numpy as np
from collections import defaultdict

def hmm_features(tokens,curr_tag,prev_tag,m):
    """Feature function for HMM that returns emit and transition features

    :param tokens: list of tokens 
    :param curr_tag: current tag
    :param prev_tag: previous tag
    :param m: index of token to be tagged
    :returns: dict of features and counts
    :rtype: dict

    """
    feat_counts = {}
    feat_counts[(curr_tag, prev_tag, TRANS)] = 1
    if m < len(tokens):
        feat_counts[(curr_tag, tokens[m], EMIT)] = 1
    return feat_counts
    

def compute_HMM_weights(trainfile,smoothing): # 6.60 bottom of p.114
    """Compute all weights for the HMM

    :param trainfile: training file
    :param smoothing: float for smoothing of both probability distributions
    :returns: defaultdict of weights, list of all possible tags (types)
    :rtype: defaultdict, list

    """
    # hint: these are your first two lines
    tag_trans_counts = most_common.get_tag_trans_counts(trainfile)
    all_tags = tag_trans_counts.keys()

    trans_weights = compute_transition_weights(tag_trans_counts, smoothing)
    for tag in all_tags:
        trans_weights[(tag, END_TAG, TRANS)] = -np.inf
    trans_weights[(END_TAG, END_TAG, TRANS)] = -np.inf
    
    word_counters = most_common.get_tag_word_counts(trainfile) # dict of counters from tag to words
    emit_weights = naive_bayes.estimate_nb_tagger(word_counters, smoothing) # defaultdict of classifier weights
    
    weights = defaultdict(float, trans_weights)
    for t,w in emit_weights.keys():
        if w != OFFSET:
            weights[(t, w, EMIT)] = emit_weights[(t,w)]
    
    return weights, all_tags


def compute_transition_weights(trans_counts, smoothing):
    """Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag,TRANS)] and weights

    """

    weights = defaultdict(float)
    all_tags = trans_counts.keys()
    all_tags.remove(START_TAG)
    all_tags.append(END_TAG)
    tot_smoothing = smoothing * len(all_tags)
    for source, counts in trans_counts.iteritems():
        total = sum(counts.values()) + (tot_smoothing)
        s_weights = defaultdict(float)
        for target in all_tags:
            s_weights[target] = (counts[target] + smoothing) / total
        logwsum = np.log(sum(s_weights.values()))
        for t, w in s_weights.iteritems():
            weights[(t, source, TRANS)] = np.log(w) - logwsum
        weights[(START_TAG, source, TRANS)] = -np.inf
    return weights
    


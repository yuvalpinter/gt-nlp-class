from gtnlplib import constants

# Deliverable 1.1
def word_feats(words,y,y_prev,m):
    """This function should return at most two features:
    - (y,constants.CURR_WORD_FEAT,words[m])
    - (y,constants.OFFSET)

    Note! You need to handle the case where $m >= len(words)$. In this case, you should only output the offset feature. 

    :param words: list of word tokens
    :param m: index of current word
    :returns: dict of features, containing a single feature and a count of 1
    :rtype: dict

    """
    fv = dict()
    fv[(y, constants.OFFSET)] = 1.0
    if m < len(words):
        fv[(y, constants.CURR_WORD_FEAT, words[m])] = 1.0
    return fv

# Deliverable 2.1
def word_suff_feats(words,y,y_prev,m):
    """This function should return all the features returned by word_feats,
    plus an additional feature for each token, indicating the final two characters.

    You may call word_feats in this function.

    :param words: list of word tokens
    :param y: proposed tag for word m
    :param y_prev: proposed tag for word m-1 (ignored)
    :param m: index m
    :returns: dict of features
    :rtype: dict

    """
    fv = word_feats(words,y,y_prev,m)
    if m < len(words):
        fv[(y, constants.SUFFIX_FEAT, words[m][-2:])] = 1.0
    return fv
    
def word_neighbor_feats(words,y,y_prev,m):
    """compute features for the current word being tagged, its predecessor, and its successor.

    :param words: list of word tokens
    :param y: proposed tag for word m
    :param y_prev: proposed tag for word m-1 (ignored)
    :param m: index m
    :returns: dict of features
    :rtype: dict

    """

    # hint: use constants.PREV_WORD_FEAT and constants.NEXT_WORD_FEAT
    fv = word_feats(words,y,y_prev,m)
    if m == 0:
        fv[(y, constants.PREV_WORD_FEAT, constants.PRE_START_TOKEN)] = 1.0
    else:
        fv[(y, constants.PREV_WORD_FEAT, words[m-1])] = 1.0
    if m < len(words) - 1:
        fv[(y, constants.NEXT_WORD_FEAT, words[m+1])] = 1.0
    elif m < len(words):
        fv[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1.0
    return fv

    
def word_feats_competitive_en(words,y,y_prev,m):
    raise NotImplementedError
    
def word_feats_competitive_ja(words,y,y_prev,m):
    raise NotImplementedError

def hmm_feats(words,y,y_prev,m):
    fv = dict()
    fv[(y, constants.PREV_TAG_FEAT, y_prev)] = 1.0
    if m < len(words):
        fv[(y, constants.CURR_WORD_FEAT, words[m])] = 1.0
    return fv

def hmm_feats_competitive_en(words,y,y_prev,m):
    raise NotImplementedError

def hmm_feats_competitive_ja(words,y,y_prev,m):
    raise NotImplementedError



from gtnlplib.constants import OFFSET

import operator
# use this to find the highest-scoring label
argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

def make_feature_vector(base_features,label):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    fv = {}
    fv[(label, OFFSET)] = 1
    for feat, count in base_features.iteritems():
        fv[(label, feat)] = count
    return fv
    
def predict(base_features,weights,labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    def score(feat_vec, weights):
        return sum([feat_vec[f] * weights[f] for f in feat_vec if f in weights])
    
    scores = {l:score(make_feature_vector(base_features,l),weights) for l in labels}
    return argmax(scores),scores

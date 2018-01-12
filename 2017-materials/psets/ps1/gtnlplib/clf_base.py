from gtnlplib.constants import OFFSET
import numpy as np

# hint! use this.
argmax = lambda x : max(x.iteritems(),key=lambda y : y[1])[0]

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

def predict_all(x,weights,labels):
    """Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    """
    y_hat = np.array([predict(x_i,weights,labels)[0] for x_i in x])
    return y_hat

def get_top_features_for_label(weights,label,k=5):
    """Return the five features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    """
    return sorted([w for w in weights.iteritems() if w[0][0] == label], key=lambda w: w[1], reverse=True)[:k]

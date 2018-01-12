import numpy as np
from collections import defaultdict
import coref

# deliverable 3.2
def mention_rank(markables,i,feats,weights):
    """ return top scoring antecedent for markable i

    :param markables: list of markables
    :param i: index of current markable to resolve
    :param feats: feature function
    :param weights: weight defaultdict
    :returns: index of best scoring candidate (can be i)
    :rtype: int

    """
    argmax = -1
    max = -np.inf
    for j in xrange(i+1):
        fs = feats(markables, j, i)
        score = sum([v * weights[f] for f,v in fs.items()])
        if score > max:
            max = score
            argmax = j
    return argmax

# deliverable 3.3
def compute_instance_update(markables,i,true_antecedent,feats,weights):
    """Compute a perceptron update for markable i.
    This function should call mention_rank to determine the predicted antecedent,
    and should make an update if the true antecedent and predicted antecedent *refer to different entities*

    Note that if the true and predicted antecedents refer to the same entity, you should not
    make an update, even if they are different.

    :param markables: list of markables
    :param i: current markable
    :param true_antecedent: ground truth antecedent
    :param feats: feature function
    :param weights: defaultdict of weights
    :returns: dict of updates
    :rtype: dict

    """
    pred_antecedent = mention_rank(markables,i,feats,weights)
    
    update = defaultdict(float)
    if pred_antecedent == true_antecedent:
        return update
    if pred_antecedent != i and markables[pred_antecedent]['entity'] == markables[true_antecedent]['entity']:
        return update

    for f,v in feats(markables, i, pred_antecedent).iteritems():
        update[f] -= v
        
    for f,v in feats(markables, i, true_antecedent).iteritems():
        update[f] += v

    return update

# deliverable 3.4
def train_avg_perceptron(markables,features,N_its=20):
    # the data and features are small enough that you can
    # probably get away with naive feature averaging

    weights = defaultdict(float)
    tot_weights = defaultdict(float)
    weight_hist = []
    T = 0.

    for it in xrange(N_its):
        num_wrong = 0
        for document in markables:
            true_ants = coref.get_true_antecedents(document)
            for i in xrange(len(document)):
                delta = compute_instance_update(document,i,true_ants[i],features,weights)
                if len(delta) > 0:
                    num_wrong += 1
                    for k,val in delta.iteritems():
                        if val == 0: continue
                        weights[k] += val
                        tot_weights[k] += (val * T)
                T += 1
        print num_wrong,

        # update the weight history
        weight_hist.append(defaultdict(float))
        for feature in tot_weights.keys():
            weight_hist[it][feature] = weights[feature] - tot_weights[feature]/T

    return weight_hist

# helpers
def make_resolver(features,weights):
    return lambda markables : [mention_rank(markables,i,features,weights) for i in range(len(markables))]

def eval_weight_hist(markables,weight_history,features):
    scores = []
    for weights in weight_history:
        score = coref.eval_on_dataset(make_resolver(features,weights),markables)
        scores.append(score)
    return scores

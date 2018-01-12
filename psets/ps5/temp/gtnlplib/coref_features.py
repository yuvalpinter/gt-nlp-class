import itertools
import coref_rules
from nltk import wordnet

# useful?
pronoun_list=['it','he','she','they','this','that']
poss_pronoun_list=['its','his','her','their']
oblique_pronoun_list=['him','her','them']
def_list=['the','this','that','these','those']
indef_list=['a','an','another']

# d3.1
def minimal_features(markables,a,i):
    """Compute a minimal set of features for antecedent a and mention i

    :param markables: list of markables for the document
    :param a: index of antecedent
    :param i: index of mention
    :returns: features
    :rtype: dict

    """
    f = dict()
    if i == a:
        f['new-entity'] = 1.0
        return f
    m_a = markables[a]
    m_i = markables[i]
    if not coref_rules.no_overlap(m_a,m_i):
        f['crossover'] = 1.0
    if coref_rules.exact_match(m_a,m_i):
        f['exact-match'] = 1.0
    if coref_rules.match_last_token(m_a,m_i):
        f['last-token-match'] = 1.0
    if coref_rules.match_on_content(m_a,m_i):
        f['content-match'] = 1.0
    return f

# deliverable 3.5
def distance_features(x,a,i,
                      max_mention_distance=10,
                      max_token_distance=10):
    """compute a set of distance features for antecedent a and mention i

    :param x: markable list for document
    :param a: antecedent index
    :param i: mention index
    :param max_mention_distance: upper limit on mention distance
    :param max_token_distance: upper limit on token distance
    :returns: feature dict
    :rtype: dict

    """
    f = dict()
    if a == i:
        return f
    ment_dist = min(i - a, max_mention_distance)
    f['mention-distance-{}'.format(ment_dist)] = 1
    tok_dist = min(x[i]['start_token'] - x[a]['end_token'], max_token_distance)
    f['token-distance-{}'.format(tok_dist)] = 1
    return f

###### Feature combiners

# deliverable 3.6
def make_feature_union(feat_func_list):
    """return a feature function that is the union of the feature functions in the list

    :param feat_func_list: list of feature functions
    :returns: feature function
    :rtype: function

    """
    def f_out(x,a,i):
        f = dict()
        for ff in feat_func_list:
            f.update(ff(x,a,i))
        return f
    return f_out

# deliverable 3.7
def make_feature_cross_product(feat_func1,feat_func2):
    """return a feature function that is the cross-product of the two feature functions

    :param feat_func1: a feature function
    :param feat_func2: a feature function
    :returns: another feature function
    :rtype: function

    """
    def f_out(x,a,i):
        f = dict()
        for (f1,v1),(f2,v2) in itertools.product(feat_func1(x,a,i).items(), feat_func2(x,a,i).items()):
            f[f1+"-"+f2] = v1 * v2
        return f
    return f_out

# deliverable 3.9

def make_bakeoff_features():
    #return make_feature_union([minimal_features, distance_features, hypernymy_features])
    return make_feature_union([minimal_features, distance_features, no_pron_match_feats, synset_match_feats])

def no_pron_match_feats(markables,a,i):
    f = dict()
    m_a = markables[a]
    m_i = markables[i]
    if coref_rules.exact_match_no_pronouns:
        f['exact-match-no-prons'] = 1.0
    return f

def wn_pos(pos):
    if pos.startswith("N"): return wordnet.NOUN
    if pos.startswith("V"): return wordnet.wordnet.VERB
    return None

def synset_match_feats(markables,a,i):
    f = dict()
    m_a = markables[a]
    m_i = markables[i]
    a_last = m_a['string'][-1].lower()
    i_last = m_i['string'][-1].lower()
    if i_last == a_last: return f
    try:
        i_synsets = set(wordnet.wordnet.synsets(i_last))
        a_synsets = set(wordnet.wordnet.synsets(a_last))
        inter = len(i_synsets.intersection(a_synsets))
        if inter > 0:
            f['matching-synsets'] = inter / len(i_synsets.union(a_synsets))
    except: pass
    return f

### NOPE
    
def hypernym_path(word, pos, level = 0, max_size = 10):
    path_sets = set()
    if level > 2:
        return path_sets
    try:
        synsets = [l.synset() for l in wordnet.wordnet.lemmas(word, wn_pos(pos))]
        for s in synsets:
            #path_sets.update([l.name().lower() for l in s.lemmas()])
            for h in s.hypernyms():
                for l in h.lemmas():
                    if l.name().lower() in path_sets: continue
                    path_sets.update(hypernym_path(l.name().lower(), pos, level + 1))
                    path_sets.add(l.name().lower())
                    if len(path_sets) > max_size: return path_sets    
    except:
        pass
    return path_sets
    #[x.lemmas()[0].name() for x in [l[0] for l in [d.synset().hypernyms() for d in wordnet.wordnet.lemmas(word)]]]

def hypernymy_features(markables,a,i):
    f = dict()
    m_a = markables[a]
    m_i = markables[i]
    h_a = m_a['string'][-1]
    h_i = m_i['string'][-1]
    a_hyps = hypernym_path(h_a, m_a['tags'][-1])
    i_hyps = hypernym_path(h_i, m_i['tags'][-1])
    if h_a.lower() in i_hyps:
        f['hypernym'] = 1.0
    if h_i.lower() in a_hyps:
        f['hyponym'] = 1.0
    return f


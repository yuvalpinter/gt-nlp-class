### Rule-based coreference resolution  ###########
# Lightly inspired by Stanford's "Multi-pass sieve"
# http://www.surdeanu.info/mihai/papers/emnlp10.pdf
# http://nlp.stanford.edu/pubs/conllst2011-coref.pdf

import nltk

# this may help
pronouns = ['I','me','mine','you','your','yours','she','her','hers','he','him','his','it','its','they','them','their','theirs','this','those','these','that','we','our','us','ours']
downcase_list = lambda toks : [tok.lower() for tok in toks]

############## Pairwise matchers #######################

def exact_match(m_a,m_i):
    """return True if the strings are identical

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if the strings are identical
    :rtype: boolean

    """
    return downcase_list(m_a['string'])==downcase_list(m_i['string'])

# deliverable 2.2
def exact_match_no_pronouns(m_a,m_i):
    """return True if strings are identical and are not pronouns

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if the strings are identical and are not pronouns
    :rtype: boolean

    """
    #return exact_match(m_a,m_i) and (len(m_a['string']) != 1 or m_a['string'][0] not in pronouns) # previous correct
    #return exact_match(m_a,m_i) and (len(m_a['string']) != 1 or m_a['string'][0].lower() not in pronouns)
    return exact_match(m_a,m_i) and (len(m_a['string']) != 1 or m_a['string'][0].lower() not in (pronouns + ['i'])) # correct but fails test

# deliverable 2.3
def match_last_token(m_a,m_i):
    """return True if final token of each markable is identical

    :param m_a: antecedent markable
    :param m_i: referent markable
    :rtype: boolean

    """
    return m_a['string'][-1].lower() == m_i['string'][-1].lower()

# deliverable 2.4
def match_last_token_no_overlap(m_a,m_i):
    """

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if final tokens match and strings do not overlap
    :rtype: boolean

    """
    return no_overlap(m_a, m_i) and match_last_token(m_a, m_i)

# deliverable 2.5
def match_on_content(m_a, m_i):
    """

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if all match on all "content words" (defined by POS tag) and markables do not overlap
    :rtype: boolean

    """
    content_tags = ['NN', 'NNP', 'NNS', 'NNPS', 'PRP', 'PRP$', 'CD']
    def content_words(mkbl):
        return [s.lower() for s,t in zip(mkbl['string'], mkbl['tags']) if t in content_tags]
    return no_overlap(m_a, m_i) and content_words(m_a) == content_words(m_i)


########## helper code

def precedes(a,b):
    return a['end_token'] <= b['start_token']

def no_overlap(a,b):
    return precedes(a, b) or precedes(b, a)

def most_recent_match(markables,matcher):
    """given a list of markables and a pairwise matcher, return an antecedent list
    assumes markables are sorted

    :param markables: list of markables
    :param matcher: function that takes two markables, returns boolean if they are compatible
    :returns: list of antecedent indices
    :rtype: list

    """
    antecedents = range(len(markables))
    for i,m_i in enumerate(markables):
        for a,m_a in enumerate(markables[:i]):
            if matcher(m_a,m_i):
                antecedents[i] = a
    return antecedents

def make_resolver(pairwise_matcher):
    """convert a pairwise markable matching function into a coreference resolution system, which generates antecedent lists

    :param pairwise_matcher: function from markable pairs to boolean
    :returns: function from markable list and word list to antecedent list
    :rtype: function

    The returned lambda expression takes a list of words and a list of markables.
    The words are ignored here. However, this function signature is needed because
    in other cases, we want to do some NLP on the words.

    """
    return lambda markables : most_recent_match(markables,pairwise_matcher)

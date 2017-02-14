# 3.1 (0.5 points)

Fill in the rest of the table below:

|      | they | can | can | fish | END |
|------|------|-----|-----|------|-----|
| Noun | -2   | -10 | -10 | -15  | n/a |
| Verb | -13  | -6  | -11 | -16  | n/a |
| End  | n/a  | n/a | n/a | n/a  | -17 |


# 4.3 (0.5 points)

Do you think the predicted tags "PRON AUX AUX NOUN" for the sentence "They can can fish" are correct? Use your understanding of parts-of-speech from the notes.

No, they are not correct. While the first "can" acts indeed as an auxiliary verb, the second "can" is a semantics-carrying verb and thus requires the tag VERB.

# 4.4 (0.5 points)

The HMM weights include a weight of zero for the emission of unseen words. Please explain:

- why this is a violation of the HMM probability model explained in the notes;
- How, if at all, this will affect the overall tagging.

A zero emission weight for any word in an observed sentence corresponds to a probability of 1 that this word is emitted by this tag. Each unseen word observed thus violates the sum_w(P)=1 constraint for all tags (the total emission probabilities is now 1 + |unseen words|).
This behaviour should not affect the overall tagging, since decoding does not require normalization. Its actual effect will be that an unseen word is assumed to be emitted from all tags with the same likelihood.

# 5.1 (1 point 4650; 0.5 points 7650)

Please list the top three tags that follow verbs and nouns in English and Japanese.

English verbs: [(u'DET', 5134), (u'ADP', 4553), (u'PRON', 4086)]
Japanese verbs: [(u'NOUN', 6236), ('--END--', 5232), (u'PUNCT', 3018)]

English nouns: [(u'PUNCT', 10100), (u'ADP', 7158), (u'NOUN', 4346)]
Japanese nouns: [(u'NOUN', 19477), (u'VERB', 12790), (u'PUNCT', 4528)]

Try to explain some of the differences that you observe, making at least two distinct points about differences between Japanese and English.

Japanese is an SOV language, meaning verbs occur after the object (which is typically a noun). Hence the high probability of the END tag (as well as PUNCT which is often clause-final as well) following a verb, whereas in English an object noun phrase (which may begin with a determiner or just a noun, such as the case with plurals) is more likely.

Japanese allows ommission of context-known pronouns. As a result, pronouns are less common and do not appear in the top-3 follow lists for neither verbs (as they do for English) or nouns.

# 6 (7650 only; 1 point)

Find an example of sequence labeling for a task other than part-of-speech tagging, in a paper at ACL, NAACL, EMNLP, EACL, or TACL, within the last five years (2012-2017). 

## List the title, author(s), and venue of the paper.

Yuval Pinter, Roi Reichart, Idan Szpektor. Syntactic Parsing of Web Queries with Question Intent. NAACL 2016

## What is the task they are trying to solve?

The general task in the paper is a joint objective of finding the segmentation points for Web queries, and individual dependency parsing for each segment. As the former can be viewed a sequence labeling task and is separately evaluated (using F1 measure), I will focus on that.

## What tagging methods do they use? HMM, CRF, max-margin markov network, something else?

The authors use (as separate models):
1. a NN language model whose token-wise scores are used for segmentation (using a threshold);
2. a fully-supervised CRF on the supplied dataset;
3. a distant-supervised CRF where training examples have been automatically extracted;
4. implicit signals from a parse tree inferred for a question automatically derived from the input query.

The labeling phases for models 1 and 4 are in essence rule-based.

## Which features do they use?

For the CRFs, they use:
* unigram form features for a window of 2 tokens before and after target;
* bigram form features for a window of 1 token before and after target;
* unigram and bigram POS features for a window of 2 tokens before and after target;
* unigram form_POS features for a window of 1 token before and after target;
* distance from beginning of query;
* distance from end of query;
* combination of distance features with form, POS, form_POS.

For the NNLM (1), word features alone are used (and the resulting score for the word).

The parser (4) is off-the-shelf and its output tree is used with post-processing rules such as "segment if two subtrees are disconnected below ROOT level".

## What methods and features are most effective?

In the segmentation task, an ensemble model which takes the union of both CRFs' segmentation decisions gets the best result. The individual CRFs tie on the entire test set, whereas the distant-supervised CRF (3) wins on two sub-sets of automatically-detectable "difficult" queries.

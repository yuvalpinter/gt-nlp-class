# Deliverable 1.3

Why do you think the type-token ratio is lower for the dev data as compared to the training data?

(Yes the dev set is smaller; why does this impact the type-token ratio?)

This is due to the power law distribution nature of word frequencies. As we add more documents, we encounter less previously-unseen words (relatively), but the token counts increase linearly, with the commonest words contributing most to the token count.


# Deliverable 3.5

Explain what you see in the scatter plot of weights across different smoothing values.

For the more common (label, word) pairs, both models assign similar probabilities as the difference in smoothing has not affected them too severly. This is the diagonal from (-10,-10) to (0,0).
For rare terms, the larger smoothing value (10) assigns more probability to the (label, word) pair, as it incorporates a larger uniform prior. The log-probabilities for these terms are thus larger (smaller in absolute value), leading to the cluster around (-20, -10) and the hockey-stick tip around (-10, -10).


# Deliverable 6.2

Now compare the top 5 features for logistic regression under the largest regularizer and the smallest regularizer.
Paste the output into ```text_answers.md```, and explain the difference. (.4/.2 points)

0.05:
((u'worldnews', u'russia'), 0.21164030663434966)
((u'worldnews', u'ukraine'), 0.19671864042167875)
((u'worldnews', u'plane'), 0.19669045608682853)
((u'worldnews', '**OFFSET**'), 0.19423893089925207)
((u'worldnews', u'russian'), 0.18417769664697603)
0.005:
((u'worldnews', u'plane'), 0.42643638297587105)
((u'worldnews', u'ai'), 0.41775992334006445)
((u'worldnews', u'russia'), 0.41724488664425463)
((u'worldnews', u'ukraine'), 0.41630247642581375)
((u'worldnews', u'russian'), 0.39469292304958775)
0.05:
((u'science', u'research'), 0.19569802904231817)
((u'science', u'study'), 0.17301368568183625)
((u'science', u'ebv'), 0.17121189726546729)
((u'science', u'corn'), 0.16426017282603228)
((u'science', '**OFFSET**'), 0.15658677257400266)
0.005:
((u'science', u'research'), 0.42934181356001561)
((u'science', u'ebv'), 0.37643188783942472)
((u'science', u'study'), 0.37304111650818067)
((u'science', u'corn'), 0.329167186084251)
((u'science', u'evolution'), 0.30833764933954172)
0.05:
((u'askreddit', u'one'), 0.11548877286935059)
((u'askreddit', u'porn'), 0.1113441262915057)
((u'askreddit', u'try'), 0.10421580417557338)
((u'askreddit', u'*'), 0.10257592733184773)
((u'askreddit', u'some'), 0.10238745374833458)
0.005:
((u'askreddit', u'porn'), 0.30525849851955028)
((u'askreddit', u'one'), 0.22567071898081303)
((u'askreddit', u'some'), 0.20847065453796543)
((u'askreddit', u'go'), 0.20444503121793078)
((u'askreddit', u'night'), 0.19895380211663097)
0.05:
((u'iama', u'!'), 0.19827293702645649)
((u'iama', '**OFFSET**'), 0.19337346185245413)
((u'iama', u'gun'), 0.12924650857780753)
((u'iama', u'request'), 0.11950431481954564)
((u'iama', u'thanks'), 0.11899834543214928)
0.005:
((u'iama', u'gun'), 0.29839751931647057)
((u'iama', u'thanks'), 0.28185368813862249)
((u'iama', u'state'), 0.2803135919431522)
((u'iama', u'marijuana'), 0.27589492788731468)
((u'iama', u'request'), 0.25325866621245496)
0.05:
((u'todayilearned', u'hr'), 0.15447256848734742)
((u'todayilearned', u'apple'), 0.11684426245347988)
((u'todayilearned', u'latin'), 0.11480421689027279)
((u'todayilearned', u'bear'), 0.083668520106146033)
((u'todayilearned', u'than'), 0.082812588711843854)
0.005:
((u'todayilearned', u'hr'), 0.36657721854272185)
((u'todayilearned', u'latin'), 0.29085767338293178)
((u'todayilearned', u'apple'), 0.28992512285814342)
((u'todayilearned', u'bear'), 0.23570760443564784)
((u'todayilearned', u'ancient'), 0.19484589233850455)

The small regularizer (0.005) tends to surface more rare words ("marijuana"; "evolution"; "ai"). They might be very indicative of the subject, but their getting a high weight contributes more to the overall weights vector than is desirable under a larger regularization constraint (0.05).
In addition, three classes surface their OFFSET feature in the 0.05 weight vectors, highlighting the importance of the most common feature of all in a constrained model.


# Deliverable 7.2

Explain the new preprocessing that you designed: why you thought it would help, and whether it did.

I added several classes of features, all of which helped performance on the dev set:
- Stemmed token: allow words with the same stem to be counted together and contribute together to their class probabilities.
- Capitalization feature: stylistic choice that can show formality difference between genres.
- Prefix and Suffix features: designed to augment the lexical information, maybe they encode structural information about the text.
- Bigram features: to capture phrases and collocations.

One type of feature did not help:
- Word length, sentence length: helpful if some topics are more verbose and/or use more sophisticated words than others (for stylistic or informativeness reasons).

# Deliverable 8

Describe the research paper that you have chosen.

- What are the labels, and how were they obtained?
- Why is it interesting/useful to predict these labels?  
- What classifier(s) do they use, and the reasons behind their choice? Do they use linear classifiers like the ones in this problem set?
- What features do they use? Explain any features outside the bag-of-words model, and why they used them.
- What is the conclusion of the paper? Do they compare between classifiers, between feature sets, or on some other dimension? 
- Give a one-sentence summary of the message that they are trying to leave for the reader.

I chose, from the NAACL 2016 Proceedings, the paper by Joty et al. entitled "Joint Learning with Global Inference for Comment Classification in Community Question Answering".
The task is determining which comments (/questions) in a question-answers thread are "good" and which are "bad". In fact, in their test set they collapse every non-"good" label from the original dataset to "bad" in order to remain with a binary classification task.
The authors employ a joint model which attempts to exploit the relationships between pairs of comments to propagate the labels accordingly. They thus examine several methods which look at a structured problem in a graph-label-propagation manner. Their state-of-the-art classifier is a Fully-Connected Conditional Random Field (FCCRF) which is like a regular CRF but conditions state factors on the data as well (so in the graphical model, the edges connecting x_i with y_i are undirected rather than directed). For the local estimation problem (each comment on its own, pre-propagation) and the factors, they use a logistic regression target function (so, softmax over labels).
Features are not described in full. They have two types: one for "node level" (meaning a single comment) that include (i) textual similarity between original question and the classified comment; (ii) boolean features for patterns such as URLs, lexicon-based positive/negative words, forum categories, stylistic markers, etc. (iii) non-textual features relating to thread meta-data. The second type is "edge level" for the relationships between comments. These include the delta for all node-level features; text similarity features between comments; the output of the node-level classifier on each node. For the textual features, there is no specific motivation for using them given, and I assume that this is the standard procedure for this type of task these days.
The paper compares these and other, baseline, classifiers on a single test set (obtained from a SemEval 2015 task). There is no testing for the composition of the feature set - it's the same for all algorithms. FCCRF wins on both major metrics - F1 on the "good" category, and Accuracy.
The message for the reader is to show the importance of acknowledging the entire set of answers given to a question (context), and to do so in conjunction with looking at each text in itself (joint learning instead of pipeline).

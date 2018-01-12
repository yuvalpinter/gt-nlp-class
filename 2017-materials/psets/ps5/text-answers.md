**Deliverable 3.8** (7650 only; 4650 optional)

To match nominals, it is often necessary to capture semantics. Find a paper (in ACL, NAACL, EACL, or TACL, since 2007) that attempts to use semantic analysis to do nominal coreference, and explain:

- What form of semantics they are trying to capture (e.g., synonymy, hypernymy, predicate-argument, distributional)
- How they formalize semantics into features, constraints, or some other preference
- How much it helps

Title: Simple Coreference Resolution with Rich Syntactic and Semantic Features
Authors: Aria Haghighi and Dan Klein
Conference: EMNLP 2009

The authors use a bootstrapping method to augment a constraint (sieve) system with semantic constraints in addition to the previous syntactic ones. They collect predicate-argument and appositival relationships, mainly between entities and their types (e.g. "Microsoft"-"company") from automatically-parsed unlabeled text corpora, filter those out for frequency, and use the obtained dictionary as a downstream filter after the syntactic components have been exhausted. As such, it is a recall-oriented method. Their system raises F1 on all metrics and test sets. For example, on the Culotta et al. 2004 test set, MUC F1 rises from 70.2 with syntactic constraints, to 79.6 with augmented semantics, indeed almost exclusively from the recall side.
NOTE: This reported F1 score might be mistaken, as the recall and precision figures compute to an F1 of *76.2*. Maybe someone should ask the authors about this?...

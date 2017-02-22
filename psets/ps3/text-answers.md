# 1.8

As you can see from the cells here, the Japanese tagger suffered more of a decrease in accuracy when moving to the test data. Why might that be?

As the lines I added to the notebook show, the Japanese training set is about 30% shorter than the English one, whereas the number of active features is about double. Holding the number of iterations and the model itself constant, these two facts lead to a sparser representation available to the Japanese model, leading to worse generalization.

# 2.3

Explain why you think suffix features are so helpful in Japanese. You may want to look at the raw data, and consult some resources that you can find from Google. (No prior Japanese knowledge is assumed!)

Japanese is an agglutinative language with morphological suffixes that are POS-specific. This means its suffixes are very indicative for part of speech, as inflected forms are common.

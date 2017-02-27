from gtnlplib import tagger_base


def kaggle_output(testfile, tagger_func, features, weights, all_tags, output_file):
    tagger = lambda words, all_tags : tagger_func(words, features, weights, all_tags)[0]
    tagger_base.apply_tagger(tagger, output_file, testfile=testfile, all_tags=all_tags, kaggle=True)
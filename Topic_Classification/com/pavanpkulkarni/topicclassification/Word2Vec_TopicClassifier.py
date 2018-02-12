import gensim, operator
from scipy import spatial
import numpy as np
from .TopicTaxonomy import topic_taxonomy

# function checks whether the input words are present in the vocabulary for the model
def vocab_check(vectors, words):
    output = list()
    for word in words:
        if word in vectors.vocab:
            output.append(word.strip())

    return output

# function calculates similarity between two strings using a particular word vector model
def calc_similarity(input1, input2, vectors):
    s1words = set(vocab_check(vectors, input1.split()))
    s2words = set(vocab_check(vectors, input2.split()))

    if len(s1words) == 0 or len(s2words) == 0:
        output = 0.0
    else:
        output = vectors.n_similarity(s1words, s2words)

    return output

# function takes an input string, runs similarity for each item in topic_taxonomy, sorts and returns top 3 results
def classify_topics(input, vectors):
    feed_score = dict()
    for key, value in topic_taxonomy.items():
        max_value_score = dict()
        for label, keywords in value.items():
            max_value_score[label] = 0
            topic = (key + ' ' + keywords).strip()
            max_value_score[label] += float(calc_similarity(input, topic, vectors))

        sorted_max_score = sorted(max_value_score.items(), key=operator.itemgetter(1), reverse=True)[0]
        feed_score[sorted_max_score[0]] = sorted_max_score[1]
    return sorted(feed_score.items(), key=operator.itemgetter(1), reverse=True)[:3]







# output1 = classify_topics('Amazon CEO and world’s richest man Jeff Bezos avoids a common', model_word2vec)
# print(output1)
#
# output2 = classify_topics('Researchers find "simple" way to hack Amazon Key', model_word2vec)
# print(output2)
#
# output3 = classify_topics('Mail carriers: USPS supervisors warn Amazon customers will get', model_word2vec)
# print(output3)
#
# output4 = classify_topics('Your New Mailman Works for Amazon', model_word2vec)
# print(output4)
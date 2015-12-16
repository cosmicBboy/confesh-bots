import logging
import pandas as pd
import os
import random
import numpy as np
from itertools import islice
from gensim.models import Word2Vec
from collections import Counter, OrderedDict

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

raw_fp = './tmp/raw/holyokecon_confessional_secrets.csv'
input1_fp = './tmp/clean/holyokecon_confessional_secrets.csv'
input2_fp = './tmp/clean/smithcon_confessional_secrets.csv'
model_fp = './tmp/model/word2vec1.wtv'
questions_fp = './data/questions-words.txt'
raw_text = pd.read_csv(raw_fp)['confession']

df = pd.read_csv(input1_fp)
text = df['clean_tokens'][df['clean_tokens'].notnull()]
text = text.apply(lambda x: x.split())

test_set = pd.read_csv(input2_fp)
test_set = test_set['clean_tokens'][test_set['clean_tokens'].notnull()]
test_set = test_set.apply(lambda x: x.split())

model = Word2Vec

model_options = {
    'min_count': 5,
    'size': 100,
    'alpha': 0.025,
    'window': 5,
    'min_count': 5,
    'max_vocab_size': None,
}

if os.path.isfile(model_fp):
    m = model.load(model_fp)
else:
    # model with default values
    m = model(text, **model_options)
    m.save(model_fp)

# set vocab
VOCAB = m.vocab.keys()

c = Counter()
for _, doc in text.iteritems():
    c.update(doc)

'''
Some useful methods
-------------------

# c.most_common(100)
print m['queer']

# vector of a single token
print m.similarity('queer', 'lesbian')

# cosine similarity of two tokens
print m.similarity('queer', 'lesbian')

# cosine similarity of two token sets
print m.n_similarity(['sleep', 'deprived'], ['stressed', 'exam'])

# most similar word
print m.most_similar('queer')

# most similar word pair
print m.most_similar(positive=['exam', 'stress'], negative=['drinking'])
print m.most_similar(positive=['gay','lesbian'], negative=['boy'])

# get words in the
print m.vocab.keys()

# log probability scores
print m.score(text, total_sentences=text.shape[0])
print m.score(test_set, total_sentences=test_set.shape[0])

# accuracy score
print m.accuracy(questions_fp)

'''

#----------------------------#
# Compute similarity in docs #
#----------------------------#

#indices generated from random.seed(42) in similarity.py
sim_indices = [68028, 2942, 29157, 24096, 77732,
               71852, 92841, 9573, 45411, 3467]

# randomly sample some sentences
random.seed(42)
choice = [random.choice(text.index) for _ in range(20)]
sample = text.loc[sim_indices][9:10]
corpus = text

print sim_indices
print sample.head()

# compute similarity scores for documents for all samples
N = 11
sim = OrderedDict()
for i, s1 in sample.iteritems():
    sim[i] = {}
    s1 = [t1 for t1 in s1 if t1 in VOCAB]
    sim[i]['query'] = s1
    results_list = []
    for j, s2 in corpus.iteritems():
        s2 = [t2 for t2 in s2 if t2 in VOCAB]
        cos_sim = m.n_similarity(s1, s2)
        if np.isnan(cos_sim).any():
            pass
        else:
            results_list.append((j, cos_sim, s2))

    # sort by score
    results_list.sort(key=lambda item: -item[1])

    #take top 10
    sim[i]['results'] = results_list[:N]

# print out results
for index, result_dict in sim.items():

    raw_query_text = raw_text.iloc[index].replace('\n', '')
    if len(result_dict['query']) == 0:
        continue
    else:
        print ""
        print  "Index: %d - Query: %s" % (index, raw_query_text)
        print ""

    for result_index, sim_score, result_text in sim[index]['results']:
        result_text = " ".join(result_text)
        raw_result_text = raw_text.iloc[result_index].replace('\n', '')
        if len(result_text) == 0 or result_index == index:
            continue
        else:
            print ("\tIndex: %d - score: %f - %s" % (result_index, sim_score,
                                                     raw_result_text))
    print ""

import logging
import pandas as pd
import random
import os

from preprocess import CLEAN_COLUMN
from gensim import corpora, models, similarities

LdaModel = models.ldamodel.LdaModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# TODO: need to actually
exp_num = 1
raw_fp = './tmp/raw/holyokecon_confessional_secrets.csv'
dict_fp = "./topic_modelling/dicts/lda_model_specs_secrets_pdd%d.dict" % exp_num
corpus_fp = "./topic_modelling/corpus/lda_model_specs_secrets_pdd%d.mm" % exp_num
model_fp = './topic_modelling/models/lda_model_specs_secrets_pdd%d.lda' % exp_num
sim_fp = './topic_modelling/similarity/lda_model_specs_secrets_pdd%d.index' % exp_num

raw_text = pd.read_csv(raw_fp)['confession']
dictionary = corpora.Dictionary.load(dict_fp)
corpus = corpora.MmCorpus(corpus_fp)
lda_model = LdaModel.load(model_fp)

input_fp = 'tmp/clean/holyokecon_confessional_secrets.csv'
df = pd.read_csv(input_fp)

text = df[CLEAN_COLUMN].astype(str)
filtered_text = text[text.apply(lambda x: True if len(x.lower().split()) > 10 else False)]
bow_text = filtered_text.apply(lambda x: x.lower().split())
bow_text = bow_text.apply(lambda x: dictionary.doc2bow(x))

if os.path.isfile(sim_fp):
    index = similarities.MatrixSimilarity.load(sim_fp)
else:
    index = similarities.MatrixSimilarity(lda_model[corpus])
    index.save(sim_fp)

# randomly sample 10 texts
random.seed(42)
doc_indices = [random.choice(filtered_text.index) for _ in range(10)]

print ""
for i in doc_indices:
    print "QUERY: {}".format(raw_text[i].replace("\n", ""))
    sims = index[lda_model[bow_text[i]]]
    # sort the similarity vector by score
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    n = 10

    for sim_index, score in sims[:n]:
        print "%f: %s" % (score, raw_text[sim_index].replace("\n", ""))
    print ""

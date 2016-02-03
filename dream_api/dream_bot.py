'''
Dream Bot

This bot parses secrets from dream.confesh.com and replies to the secret with
a top sentence to interpret the dream based on this process:

Obtain Secrets from dreams.confesh.com
1.1 Connect to Mongo database, create cursor stream for secret.
1.2 Read in complete dream corpus.

Train a Word2Vec Model
2.1 Concatenate dream secrets with dream corpus definition entries.
2.2 Train a Word2Ven model using Gensim

Matching Dream Interpretations with Secrets
3.1 tokenize the secret (unigrams and bigrams), remove stopwords, filter freq.
3.2 For each secret token, find dream corpus entries that contain the token.
3.3 For each match, get the dream definition and sentence tokenize the
    definition string.
3.4 Take the original tokenized secret and compute rank-ordered similarity of
    all matching sentences using Word2Vec model.
3.5 Select top ranked sentence and insert it to the top of the comments field
    of the corresponding secret MongoDB document.

SCHEMA: mongo.confesh-db.confession
-----------------------------------

{
  "_id": ObjectId("55797807e4b0a796c3fc8c51"),
  "_class": "com.confesh.server.model.Confession",
  "communities": [
    "community1"
  ],
  "text": "Not a letter.",
  "timestamp": ISODate("2015-06-11T11:59:03.167Z"),
  "lastCommented": ISODate("2015-06-11T12:22:47.667Z"),
  "isLetter": false,
  "comments": [
    {
      "_id": "8877fa98-d312-4a1d-9efa-48bb371a080b",
      "text": "Comment 1",
      "timestamp": ISODate("2015-06-11T12:22:43.836Z")
    },
    {
      "_id": "d0f88972-8e2f-4930-8463-e08487f6dd0d",
      "text": "Comment 2",
      "timestamp": ISODate("2015-06-11T12:22:47.667Z"),
      "reports": [
        {
          "type": "ANONYMOUS",
          "reason": "offensive!",
          "timestamp": ISODate("2015-06-11T12:23:10.642Z")
        }
      ],
      "status": "HIDDEN"
    }
  ],
  "numberOfComments": 2,
  "views": 0,
  "md5": "4e80eb4e939f6769bb51142cd434a7e7"
}
'''

import logging
import time
import os
import pandas as pd
import re
import string
from collections import Counter, OrderedDict
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from pymongo import MongoClient

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

stop_words = set(stopwords.words('english'))

def read_mongo(domain, port, db_name, coll_name):
    return MongoClient(domain, port)[db_name][coll_name]

def read_dream_corpus(collection, query, projection):
    return scrub_cursor(collection.find(query, projection=projection))

def scrub_cursor(cursor, text_field='text'):
    return [scrub_text(c['text']) for c in cursor]

def scrub_text(text, len_thres=3):
    '''
    - find text field,
    - lowercase
    - remove numbers
    - remove punctuation
    '''
    text = re.compile('[0-9]').sub("", text.lower())
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub("", text)
    return [w for w in word_tokenize(text)
            if w not in stop_words
            and w != 'dream'
            and len(w) > len_thres]

def train_model(model, scrubbed_cursor, save_fp, **model_opts):
    m = model(scrubbed_cursor, **model_opts)
    m.save(save_fp)
    return m

def load_model(model, fp):
    return model.load(fp)

def sent_word_tokenize(text):
    sentences = sent_tokenize(text)
    return [word_tokenize(s) for s in sentences]

def parse_secret():
    pass

def match_tokens_to_dreams():
    pass

def prep_dream_defitions():
    pass

def rank_interpretations():
    pass

def insert_interpretation():
    pass


if __name__ == "__main__":
    start = time.time()
    collection = read_mongo('confesh.com', 27017, 'confesh-db', 'confession')
    confesh_corpus = read_dream_corpus(collection, {'communities': 'dreams'},
                                       projection={'text': 1, '_id': 0})
    print "Time taken to read mongo: {}".format(time.time() - start)

    model_options = {
        'min_count': 2,
        'size': 100,
        'alpha': 0.025,
        'window': 5,
        'min_count': 5,
        'max_vocab_size': None, }

    start = time.time()
    model_fp = 'dream_api/models/dream_bot_v0.1'
    d_fp = './data/dream_corpus_complete.csv'
    dream_df = pd.read_csv(d_fp)
    dream_corpus = [scrub_text(d) for d in dream_df['definitions']]

    confesh_dream_corpus = dream_corpus + confesh_corpus

    if os.path.isfile(model_fp):
        m = load_model(Word2Vec, model_fp)
    else:
        m = train_model(Word2Vec, confesh_dream_corpus, model_fp,
                        **model_options)
    print "Time taken for model training: {}".format(time.time() - start)

    VOCAB = m.vocab.keys()
    c = Counter()
    for doc in confesh_dream_corpus:
        c.update(doc)

    sim = []

    start = time.time()
    # query = 'I dream of giving birth to a beautiful baby girl'
    query = 'I dream of poop and snake'
    query_tokens = [w for w in scrub_text(query) if w in VOCAB]
    print query
    dream_subset_raw = dream_df[dream_df['vocab']\
        .apply(lambda x: True if x in query_tokens else False)]['definitions']

    # Number of sentences to include in the subset corpus.
    # this is because some interpretations are long and after a few sentences
    # the sentence recommendations are a little noise
    sentence_threshold = 2

    dream_subset_sents = dream_subset_raw.apply(sent_word_tokenize).tolist()
    flat_subset_sents = [" ".join(sent) for dream in dream_subset_sents
                         for sent in dream[:sentence_threshold]]
    dream_subset_sents = [scrub_text(sent) for sent in flat_subset_sents]

    for i, d in enumerate(dream_subset_sents):
        dream = [t1 for t1 in d if t1 in VOCAB]
        if len(dream) == 0:
            continue
        else:
            sim.append([m.n_similarity(query_tokens, dream), i])

    sim.sort(key=lambda item: -item[0])
    top_hits = [tup[1] for tup in sim[:5]]
    print query
    print top_hits
    count = 1
    for i, hit in pd.Series(flat_subset_sents).loc[top_hits].iteritems():
        print "{}: {}".format(count, hit)
        count += 1
    print "Time taken for query: {}".format(time.time() - start)

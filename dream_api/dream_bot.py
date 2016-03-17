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
import numpy as np
import re
import json
import string
import mongo_creds as creds
import inflect
import uuid
import sys
from os import path
from bson.objectid import ObjectId
from datetime import datetime
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from argparse import ArgumentParser
from collections import Counter, OrderedDict
from gensim.models import Word2Vec
from pymongo import MongoClient
from confesh_api import fetch_auth_token, post_comment

DATETIME_THRES = datetime(2015, 12, 31, 0, 00, 00, 000000)

ps = PorterStemmer()
inflect_engine = inflect.engine()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)

stop_words = set(stopwords.words('english'))
remove_regex_in_query = [
    'dreambot test',
    'what does it mean',
    'meaning',
    'i mean',
    'i saw',
    'dreams',
    'dreamt',
    'dream',
    'dreaming'
]

mongo_comment_schema = OrderedDict({
    u'text': None,
    u'_id': None,
    u'timestamp': None,
})


def fetch_collection(domain, port, db_name, coll_name):
    return MongoClient(domain, port)[db_name][coll_name]


def read_dream_corpus(collection, query, projection=None):
    return scrub_cursor(collection.find(query, projection=projection))


def generate_dream_collection_for_insert(collection, query):
    return collection.find(query)


def scrub_cursor(cursor, text_field='text'):
    return [scrub_text(c['text']) for c in cursor]


def scrub_text(text, len_thres=2):
    '''
    - find text field,
    - lowercase
    - remove numbers
    - remove punctuation
    '''
    text = re.compile('[0-9]').sub(" ", text.lower())
    text = re.compile('[%s\.]' % re.escape(string.punctuation)).sub(" ", text)
    for regex in remove_regex_in_query:
        text = re.compile(regex).sub(" ", text)
    return [w for w in word_tokenize(text)
            if w not in stop_words
            and w != 'dream'
            and len(w) > len_thres]


def train_model(model, scrubbed_cursor, save_fp, **model_opts):
    m = model(scrubbed_cursor, **model_opts)
    m.save(save_fp)
    params_fp = "{}{}".format(model_fp, ".json")
    with open(params_fp, 'wb') as f:
        f.write(json.dumps(model_opts, indent=4))
    return m


def load_model(model, fp):
    return model.load(fp)


def sent_word_tokenize(text):
    sentences = sent_tokenize(text)
    return [word_tokenize(s) for s in sentences]


def create_query_tokens(query, vocab):
    query_tokens = [w for w in scrub_text(query)]

    # add stemmed tokens
    stemmed_tokens= [ps.stem(w) for w in query_tokens]

    # add pluralized stemmed tokens
    plurals = [inflect_engine.plural(w) for w in stemmed_tokens]
    tokens = set(query_tokens + stemmed_tokens + plurals)
    return [t for t in tokens if t in vocab]


def interpret_dream(query, vocab):
    query_tokens = create_query_tokens(query, vocab)
    print query_tokens
    dream_subset_sents = match_query_to_dreams(query_tokens,
                                               dream_df, vocab)
    top_hits = prep_dream_definitions(dream_subset_sents)
    return format_dream_interpretation(top_hits['interpretations'].tolist(),
                                       query_tokens)


def match_query_to_dreams(query_tokens, dream_df, vocab):

    # Matching logic:
    # -----------------
    # 1. Subset dream definitions by selecting only entries whose 'vocab' term
    #    is in the query_tokens list.

    dream_match = dream_df[dream_df['vocab']\
        .apply(lambda x: True if x in query_tokens else False)].reset_index()

    # Compute modifier tokens:
    # modifier tokens are defined as tokens that are in the query tokens
    # but are not in the list of vocab keywords.
    vocab_tokens = dream_match['vocab'].unique().tolist()
    plurals = [inflect_engine.plural(w) for w in vocab_tokens]
    vocab_tokens = vocab_tokens + plurals
    modifier_tokens = [t for t in query_tokens
                       if t not in vocab_tokens]

    dream_match = dream_match.groupby('vocab').apply(
        lambda x: _match_modifier_tokens(x, modifier_tokens))
    dream_match.loc[:, 'sim'] = dream_match['interpretations'].apply(
        lambda x: _compute_similarity(x, query_tokens, vocab))
    return dream_match

def _match_modifier_tokens(dream_group, modifier_tokens):
    dream_group.loc[:, 'index'] = range(dream_group.shape[0])
    dream_interp_tmp = dream_group[dream_group['index'] > 0]

    if len(modifier_tokens) == 0:
        i = [0]
    else:
        modifier_re = '|'.join(modifier_tokens)
        dream_interp_tmp = dream_interp_tmp[
            dream_interp_tmp['interpretations'].str.contains(modifier_re)]
        i = [0] + dream_interp_tmp['index'].tolist()
    return dream_group.iloc[i]


def _compute_similarity(dream, query_tokens, vocab):
    dream_tokens = [t1 for t1 in word_tokenize(dream) if t1 in vocab]
    if len(dream) == 0:
        return None
    else:
        return m.n_similarity(query_tokens, dream_tokens)


def prep_dream_definitions(dream_match_df):

    # Group each row in the interpretation matches by vocab
    # - sorted by index
    # - select the first interpretation plus the top interpretation
    #   after the first interpretation.
    # - group vocab-level interpretations together and sort those groups
    #   by average similarity to the query.
    dream_groups = dream_match_df.groupby('vocab')
    ranked_interpretations = dream_groups.apply(rank_interpretation)
    top_dreams = ranked_interpretations.reset_index(drop=True)
    top_dreams.loc[:, 'interpretations'] = postprocess_dreams(
        top_dreams['interpretations'].tolist())
    return top_dreams

def rank_interpretation(dream_group, n=1):
    '''Ranks interpretations within a dream entry

    Input: dataframe of dream interpretations
    Output: dataframe of top n dream interpretations ranked by sentence
            similarity

    Heuristic:
    - Always include the first interpretation.
    - Among the subsequent interpretations, rank by similarity score
    - Select top n interpretations to include in the dream definition
    '''
    # print dream_group
    dream_group['index'] = range(dream_group.shape[0])
    dream_interp_tmp = dream_group[dream_group['index'] > 0]\
        .sort_values('sim', ascending=False)
    i = [0] + dream_interp_tmp['index'].iloc[:n].tolist()
    return dream_group.iloc[i]


def postprocess_dreams(dream_interp_list):
    d_list = [sent.split('.') for sent in dream_interp_list]
    d_list = [[s for s in sent if s != ""] for sent in d_list]
    d_list = map(
        lambda sent_list: [format_dream_string(s) for s in sent_list], d_list)
    return [" ".join(d) for d in d_list]


def format_dream_string(text):
    text = text.strip()
    text = text.capitalize()
    text = "{}.".format(text)
    text = re.compile(r'\ ([%s])' % re.escape(string.punctuation)).sub(r"\1", text)
    return text


def format_dream_interpretation(dream_list, tokens):
    '''
    Create interpretation for production-ready format
    - Each interpretation has a paragraph
    - The hashtag tokens are at the very end on its own line

    For example:
    Interpretation 1 is about foo

    Interpretation 2 is about bar

    Interpretation 3 is about baz

    #foo #bar #baz
    '''
    interpretation = "\n\n".join(["{}".format(r) for r in dream_list])
    cite_str = "http://www.dreammoods.com/"
    hashtags = " ".join(["#{}".format(t) for t in set(tokens)])
    formatted_dream = "\n\n".join([interpretation, hashtags, cite_str])
    return "!dreambot! " + formatted_dream


def write_to_dream_log(fp, secret_id):
    if path.isfile(fp):
        mode = "a"
    else:
        mode = "wb"
    with open(fp, mode) as f:
        f.write("{}\n".format(secret_id))

def read_dream_log(fp):
    with open(fp, 'rb') as f:
        return [l.replace('\n', '') for l in f.readlines()]


def dream_passes_filter(post_object, datetime_thres=DATETIME_THRES):
    if post_object['timestamp'] < DATETIME_THRES:
        return False
    elif any([c.get('avatar', None)
         for c in post_object.get('comments', [dict()]) ]):
        return False
    else:
        return True

if __name__ == "__main__":

    parser = ArgumentParser(description='A CLI tool for DreamBot')
    parser.add_argument('-db', help='name of db')
    parser.add_argument('-c', help='collection name')
    parser.add_argument('-m', help='model filepath')
    parser.add_argument('-dr', help='dream corpus filepath')
    parser.add_argument('--id_log', help='filepath to log to posts that are' +
        'already implemented')

    args = parser.parse_args()

    start = time.time()
    collection = fetch_collection(creds.domain, creds.port,
                                  args.db, args.c)
    confesh_corpus = read_dream_corpus(collection, {'communities': 'dreams'},
                                       projection={'text': 1, '_id': 0})
    dream_test_corpus = generate_dream_collection_for_insert(
        collection, {'communities': 'bots'})

    print("Time taken to read mongo: {}".format(time.time() - start))

    model_options = {
        "alpha": 0.025,
        "trim_rule": None,
        "workers": 1,
        "min_alpha": 0.0001,
        "negative": 0,
        "iter": 20,
        "sample": 0,
        "window": 5,
        "seed": 1,
        "hs": 1,
        "max_vocab_size": None,
        "min_count": 1,
        "size": 20,
        "sg": 0,
        "cbow_mean": 0,
        "null_word": 0
    }

    start = time.time()
    model_fp = args.m
    d_fp = args.dr
    log_fp = args.id_log
    dream_df = pd.read_csv(d_fp)
    dream_corpus = [scrub_text(d) for d in dream_df['interpretations']]

    confesh_dream_corpus = dream_corpus + confesh_corpus

    # if file exists, load the file
    if os.path.isfile(model_fp):
        m = load_model(Word2Vec, model_fp)
    else:
        m = train_model(Word2Vec, confesh_dream_corpus, model_fp,
                        **model_options)

    print(
        "Time taken for model training: {}".format(time.time() - start))

    VOCAB = m.vocab.keys()
    c = Counter()
    for doc in confesh_dream_corpus:
        c.update(doc)

    sim = []

    start = time.time()


    # read dream log file
    if path.isfile(log_fp):
        dreams_already_interpreted = read_dream_log(log_fp)
    else:
        dreams_already_interpreted = []
    print('Dreams already interpreted:')
    print(dreams_already_interpreted)

    # get auth token to post comments to confesh
    auth_token = fetch_auth_token()

    for post in dream_test_corpus:
        secret_id = str(post['_id'])

        # logic for logging
        if secret_id in dreams_already_interpreted:
            continue
        else:
            print('Logging dream: {}'.format(secret_id))
            # write_to_dream_log(log_fp, secret_id)

        # logic for interpretation and commenting
        if dream_passes_filter(post):
            print post['text']
            try:
                interpretation = interpret_dream(post['text'], VOCAB)
                print interpretation
                post_comment(secret_id, auth_token, interpretation)
            except Exception as e:
                print e
                pass

    print("Time taken for query: {}".format(time.time() - start))

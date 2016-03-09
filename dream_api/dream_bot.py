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
from argparse import ArgumentParser
from collections import Counter, OrderedDict
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
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
    return [t for t in set(query_tokens + stemmed_tokens + plurals)
            if t in vocab]


def interpret_dream(query, vocab):
    query_tokens = create_query_tokens(query, vocab)
    dream_subset_sents = match_query_to_dreams(query_tokens,
                                               dream_df, vocab)
    top_hits = prep_dream_definitions(dream_subset_sents)
    return format_dream_interpretation(top_hits['definition'].tolist(),
                                       query_tokens)


def match_query_to_dreams(query_tokens, dream_df, vocab):

    # Matching logic:
    # -----------------
    # 1. Subset dream definitions by selecting only entries whose 'vocab' term
    #    is in the query_tokens list
    # 2. Tokenize the sentences in dream definitions
    dream_df = dream_df[dream_df['vocab']\
        .apply(lambda x: True if x in query_tokens else False)]
    dream_df.loc[:, 'sents'] = dream_df['definitions']\
        .apply(sent_word_tokenize)

    dream_match_index = [(i, j, " ".join(s)) for i, sent_list
                         in dream_df['sents'].iteritems()
                         for j, s in enumerate(sent_list)]

    dream_match = pd.DataFrame(dream_match_index,
                               columns=['index', 'sent_num', 'definition'])
    dream_match.loc[:, 'sim'] = dream_match['definition']\
        .apply(lambda x: compute_similarity(x, query_tokens, vocab))
    return dream_match


def compute_similarity(dream, query_tokens, vocab):
    dream_tokens = [t1 for t1 in word_tokenize(dream) if t1 in vocab]
    if len(dream) == 0:
        return None
    else:
        return m.n_similarity(query_tokens, dream_tokens)


def prep_dream_definitions(dream_match_df, top_n=5,
                           sent_thres=1, sent_append_num=3):
    top_dreams = dream_match_df[dream_match_df['sent_num'] < sent_thres]
    top_dreams = top_dreams.sort_values('sim', ascending=False)
    top_dreams = top_dreams.reset_index(drop=True)
    top_index = [i for i in top_dreams.index if i in range(top_n)]
    top_dreams = top_dreams.iloc[top_index]
    top_dreams.loc[:, 'definition'] = top_dreams.apply(
        lambda row: append_more_sentences(
            row, dream_match_df, sent_append_num), axis=1
        )
    top_dreams.loc[:, 'definition'] = postprocess_dreams(
        top_dreams['definition'].tolist())
    return top_dreams


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
    hashtags = " ".join(["#{}".format(t) for t in set(tokens)])
    formatted_dream = "\n\n".join([interpretation, hashtags])
    return "!dreambot! " + formatted_dream


def append_more_sentences(top_dream_row, dream_df, sent_append_num):
    index = top_dream_row['index']
    i_start = top_dream_row['sent_num']
    i_end = i_start + sent_append_num

    # conditions
    c1 = (dream_df['sent_num'] >= i_start)
    c2 = (dream_df['sent_num'] < i_end)
    c3 = (dream_df['index'] == index)
    append_sents = dream_df[c1 & c2 &c3]['definition'].tolist()
    return " ".join(append_sents)


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
    dream_corpus = [scrub_text(d) for d in dream_df['definitions']]

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
            write_to_dream_log(log_fp, secret_id)

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

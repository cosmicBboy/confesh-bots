'''Module for Generating Recommendations for Secrets

Given a Word2Vec Model,
1. Read secret text with ids
2. Create n x m cosine similarity matrix
   2.1 Where n are doc ids of bumped posts
   2.2 Where m are doc ids of all posts
3. For each bumped post
3. Serialize model to S3
'''

import logging
import pandas as pd
import numpy as np
import mongo_creds as creds
import json
import sys
import smart_open as so
from time import sleep
from bitly_utils import shorten_secret_url
from collections import OrderedDict
from argparse import ArgumentParser
from gensim.models import Word2Vec
from gensim import similarities
from stream_mongo import MongoStreamer
from preprocessor import TextPreprocessor
from s3_utils import create_model_key, BitlyS3Cacher

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)

tp = TextPreprocessor()
bitly_cacher = BitlyS3Cacher()

THREAD_BOT_MSG = "You bumped this post! Here are more like this one"
THREAD_BOT_CODE = '!threadbot!'


def preprocess_recommendations(rec_df):
    '''Prepares similarity recommendation dataframe for Confesh API POST call
    '''
    target_doc_groups = rec_df.groupby('recommend_doc')
    preprocessed_recs = target_doc_groups.apply(_agg_target_doc_group)\
        .reset_index(level=0)\
        .rename(columns={0: 'recommendations'})

    return preprocessed_recs.reset_index(level=0)


def _agg_target_doc_group(target_doc_group):
    '''Aggregates and formats a group of recommendations for a target document
    '''
    df = target_doc_group[['query_match', 'q_doc_id']]
    formatted_df = df.apply(_agg_target_doc_group_row, axis=1)
    return _format_message(formatted_df)


def _format_message(formatted_rec_list):
    '''Helper function to format a group of recommendations for a target document
    '''
    formatted_rec_string = '\n\n'.join(formatted_rec_list)
    return "{} {}\n\n{}".format(
        THREAD_BOT_CODE, THREAD_BOT_MSG, formatted_rec_string)


def _agg_target_doc_group_row(target_doc_group_row):
    '''Adds url link to a single recommendation document
    '''
    query_text = target_doc_group_row['query_match']
    short_url = _fetch_short_url(target_doc_group_row['q_doc_id'])
    rec_strings = _format_recommendation(query_text, short_url)
    return rec_strings


def _format_recommendation(query_match_text, short_url, max_text_len=100):
    '''Helper function to format the recommendation document string
    '''
    if len(query_match_text) > max_text_len:
        query_match_text = query_match_text[:max_text_len]
    if len(query_match_text) >= 3 and query_match_text[-3:] != '...':
        query_match_text = query_match_text + '...'
    return "{}<a href=\"{}\" target=\"_blank\"> read more</a>".format(
        query_match_text, short_url)


def _fetch_short_url(secret_id):
    '''Fetches the short_url of the specified secret_id

    Uses BitlyS3Cacher to fetch bitly data from s3 if it's cached. Otherwise,
    the cacher sends a GET request to bitly to grab the data and caches it
    for future use.
    '''
    try:
        return bitly_cacher.fetch_bitly_data(secret_id)['url']
        sleep(1)
    except:
        logging.warning("HEY THERE'S AN ERROR HERE")
        logging.warning("secret_id: {}".format(secret_id))
        pass


class Word2VecRecommender(object):

    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.vocab = self.model.vocab.keys()

    def load_model(self, model_name):
        model_key = create_model_key(model_name, 'model', 'w2v')
        return Word2Vec.load(model_key)

    def compute_sim_matrix(self, rec_docs, query_docs):
        '''Transforms eligible documents to Word2Vec space
        '''
        r_dict = {d['id']: d['text'] for d in rec_docs}
        r_doc_ids = [d['id'] for d in rec_docs
                     for _ in range(len(query_docs))]

        q_dict = {d['id']: d['text'] for d in query_docs}
        q_doc_ids = [d['id'] for d in query_docs] * len(rec_docs)

        df = pd.DataFrame({'r_doc_id': r_doc_ids,
                           'q_doc_id': q_doc_ids})
        drop_rows = df.apply(
            lambda row: self._rdoc_id_equals_qdoc_id(
                row['r_doc_id'], row['q_doc_id']),
            axis=1)
        df = df[~drop_rows]

        # compute similarity
        df.loc[:, 'sim'] = df.apply(
            lambda row: self._compute_sim(
                tp.preprocess(r_dict[row['r_doc_id']]),
                tp.preprocess(q_dict[row['q_doc_id']])),
            axis=1)
        df = df[df['sim'].notnull()]
        df = self._rank_sim(df)

        # formatting recommandation match text
        df.loc[:, 'recommend_doc'] = df['r_doc_id'].apply(
            lambda r_id: r_dict[r_id])
        df.loc[:, 'query_match'] = df['q_doc_id'].apply(
            lambda q_id: q_dict[q_id])
        return df

    def _compute_sim(self, rec_doc, query_doc):
        r_doc = [t for t in rec_doc if t in self.vocab]
        q_doc = [t for t in query_doc if t in self.vocab]
        if not r_doc or not q_doc:
            return np.nan
        return self.model.n_similarity(r_doc, q_doc)

    def _rank_sim(self, sim_df):
        grp = sim_df.groupby('r_doc_id')
        return grp.apply(self._rank_agg_func).reset_index(drop=True)

    def _rank_agg_func(self, group, n=5):
        grp = group.sort_values('sim', ascending=False)
        return grp.iloc[:n]

    def _rdoc_id_equals_qdoc_id(self, r_doc_id, q_doc_id):
        if r_doc_id == q_doc_id:
            return True
        else:
            return False

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
from collections import OrderedDict
from argparse import ArgumentParser
from gensim.models import Word2Vec
from gensim import similarities
from stream_mongo import MongoStreamer
from preprocessor import TextPreprocessor
from s3_utils import create_model_key

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)

tp = TextPreprocessor()

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

'''Entry point for running thread_bot pipeline

The thread_bot pipeline defines three different community parameters:
- The train community specifies the data source for training a Word2Vec model
- The target community specifies where to look for bumped posts to recommend
- The query community specifies the source for documents to recommend


'''

import os
import pandas as pd
from datetime import datetime
from stream_mongo import MongoStreamer
from preprocessor import TextPreprocessor
from model_builder import Word2VecModelBuilder
from model_recommender import Word2VecRecommender, preprocess_recommendations
from argparse import ArgumentParser
from s3_utils import model_exists
from confesh_api import fetch_auth_token, post_comment
import cStringIO
import mongo_creds as creds


DATETIME_THRES = datetime(2016, 03, 15, 0, 00, 00, 000000)


if __name__ == "__main__":
    parser = ArgumentParser(description='Preprocessing Layer for Confesh Bots')
    parser.add_argument('-db', help='database name', default='confesh-db')
    parser.add_argument('-cl', help='collection name', default='confession')
    parser.add_argument('-m', help='model name', default='model3')
    parser.add_argument('--train_community', default='www',
                        help='community name of train dataset to train model')
    parser.add_argument('--target_community', default='bots',
                        help='community name of the target community')
    parser.add_argument('--query_community', default='www',
                        help='community name of query community')

    args = parser.parse_args()
    stream = MongoStreamer(creds.domain, creds.port, args.db, args.cl)
    tp = TextPreprocessor()
    model_name = args.m
    train_cm = args.train_community
    target_cm = args.target_community
    query_cm = args.query_community

    # Grab auth token
    # TODO: this should probaby be in confesh_api
    auth_token_fp = './tmp/auth_token.txt'
    if os.path.isfile(auth_token_fp):
        with open(auth_token_fp, 'rb') as fp:
            auth_token = fp.read()
    else:
        auth_token = fetch_auth_token(community)
        with open(auth_token_fp, 'wb') as fp:
            fp.write(auth_token)

    params = {
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

    # # BUILD MODEL
    if model_exists(model_name):
        print 'Model already exists... skipping model-building step'
    else:
        train_docs = [d for d in
                      stream.iterate_secrets_comments(train_cm, limit=0)]
        train_doc_ids = [str(d['id']) for d in train_docs]

        w2v_builder = Word2VecModelBuilder(params)
        w2v_builder.fit(train_docs)
        w2v_builder.save_model(model_name, train_doc_ids)

    # # COMPUTE RECOMMENDATIONS
    # # This section subsets the confesh database to find
    # # candidates to act on.
    target_docs = [d for d in stream.iterate_secrets(target_cm, limit=0)
                   if not d['contains_threadbot_post']
                   and (d['bumped'] == True)]
    query_docs = [d for d in stream.iterate_secrets(query_cm, limit=0)
                  if not d['hidden'] and
                  not d['contains_threadbot_post']]
    print 'TARGET DOCS:', len(target_docs)
    print 'QUERY DOCS:', len(query_docs)

    if len(target_docs) == 0:
        raise ValueError('There are no target docs!')

    # LOAD MODEL
    w2v_rec = Word2VecRecommender(model_name)

    # PROCESS RECOMMENDATIONS
    recommend_file = './tmp/recommendations.csv'
    sim = w2v_rec.compute_sim_matrix(target_docs, query_docs)
    sim.to_csv(recommend_file)

    processed_rec_fp = './tmp/processed_recommendations.csv'
    processed_recs = preprocess_recommendations(sim, query_cm)
    processed_recs.to_csv(processed_rec_fp, index=False)

    for _, r in processed_recs.iterrows():
        secret_id = r['r_doc_id']
        recommendations = r['recommendations']
        post_comment(target_cm, secret_id, auth_token, recommendations)

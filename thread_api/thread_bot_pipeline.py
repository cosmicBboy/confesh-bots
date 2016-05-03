'''Entry point for running thread_bot pipeline
'''

import os
import pandas as pd
from datetime import datetime
from stream_mongo import MongoStreamer
from preprocessor import TextPreprocessor
from model_builder import Word2VecModelBuilder
from model_recommender import Word2VecRecommender, preprocess_recommendations
from argparse import ArgumentParser
from confesh_api import fetch_auth_token, post_comment
import cStringIO
import mongo_creds as creds


DATETIME_THRES = datetime(2016, 03, 15, 0, 00, 00, 000000)


if __name__ == "__main__":
    parser = ArgumentParser(description='Preprocessing Layer for Confesh Bots')
    parser.add_argument('-db', help='database name', default='confesh-db')
    parser.add_argument('-cl', help='collection name', default='confession')
    parser.add_argument('-cm', help='community name', default='bots')

    args = parser.parse_args()
    stream = MongoStreamer(creds.domain, creds.port, args.db, args.cl)
    tp = TextPreprocessor()
    community = args.cm

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
    # train_docs = [d for d in stream.iterate_secrets_comments(community, limit=0)]
    # train_doc_ids = [str(d['id']) for d in train_docs]

    # w2v_builder = Word2VecModelBuilder(params)
    # w2v_builder.fit(train_docs)
    # w2v_builder.save_model('model1', train_doc_ids)

    # # COMPUTE RECOMMENDATIONS
    # # This section subsets the confesh database to find
    # # candidates to act on.
    all_docs = [d for d in stream.iterate_secrets(community, limit=0)]
    recommend_docs = [d for d in all_docs if not d['contains_threadbot_post']
                      and (d['bumped'] == True)]
    query_docs = [d for d in all_docs if not d['hidden'] and
                  not d['contains_threadbot_post']]
    print 'ALL DOCS:', len(all_docs)
    print 'RECOMMEND DOCS:', len(recommend_docs)
    print 'QUERY DOCS:', len(query_docs)

    # LOAD MODEL
    w2v_rec = Word2VecRecommender('model1')

    # PROCESS RECOMMENDATIONS
    recommend_file = './tmp/recommendations.csv'
    sim = w2v_rec.compute_sim_matrix(recommend_docs, query_docs)
    sim.to_csv(recommend_file)

    processed_rec_fp = './tmp/processed_recommendations.csv'
    processed_recs = preprocess_recommendations(sim)
    processed_recs.to_csv(processed_rec_fp, index=False)

    for _, r in processed_recs.iterrows():
        secret_id = r['r_doc_id']
        recommendations = r['recommendations']
        post_comment(community, secret_id, auth_token, recommendations)

'''Entry point for running thread_bot pipeline
'''

from stream_mongo import MongoStreamer
from preprocessor import TextPreprocessor
from model_builder import Word2VecModelBuilder
from model_recommender import Word2VecRecommender
from argparse import ArgumentParser
import cStringIO
import mongo_creds as creds


if __name__ == "__main__":
    parser = ArgumentParser(description='Preprocessing Layer for Confesh Bots')
    parser.add_argument('-db', help='database name', default='confesh-db')
    parser.add_argument('-cl', help='collection name', default='confession')
    parser.add_argument('-cm', help='community name', default='www')

    args = parser.parse_args()
    stream = MongoStreamer(creds.domain, creds.port, args.db, args.cl)
    tp = TextPreprocessor()

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

    # train_docs = [d for d in stream.iterate_secrets_comments(args.cm, limit=0)]
    # train_doc_ids = [str(d['id']) for d in train_docs]

    # w2v_builder = Word2VecModelBuilder(params)
    # w2v_builder.fit(train_docs)
    # w2v_builder.save_model('model1', train_doc_ids)

    all_docs = [d for d in stream.iterate_secrets(args.cm, limit=0)]
    recommend_docs = [d for d in all_docs if not d['contains_threadbot_post']
                      and (d['bumped'] == True)]
    query_docs = [d for d in all_docs if not d['hidden'] and
                  not d['contains_threadbot_post']]
    print 'ALL DOCS:', len(all_docs)
    print 'RECOMMEND DOCS:', len(recommend_docs)
    print 'QUERY DOCS:', len(query_docs)
    w2v_rec = Word2VecRecommender('model1')
    sim = w2v_rec.compute_sim_matrix(recommend_docs, query_docs)
    sim.to_csv('./tmp/recommendations.csv')
'''Module for Building a Model

Train a Word2Vec Model based on secret and comment text on www.confesh.com
1. Read secret and comment text
2. Train a Word2Vec model
3. Serialize model to S3
'''

import logging
import pandas as pd
import mongo_creds as creds
import json
import sys
import smart_open as so
from collections import OrderedDict
from argparse import ArgumentParser
from gensim.models import Word2Vec
from stream_mongo import MongoStreamer
from preprocessor import TextPreprocessor
from s3_utils import create_model_key

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)

tp = TextPreprocessor()


class Word2VecModelBuilder(object):

    def __init__(self, params):
        self.model = Word2Vec
        self.params = params

    def fit(self, train_docs):
        token_list = [tp.preprocess(d['text']) for d in train_docs]
        self.model = self.model(token_list, **self.params)

    def save_model(self, model_name, document_ids):
        s3_keys = self._get_s3_keys(model_name)
        self.model.save(s3_keys['model'])
        with so.smart_open(s3_keys['params'], 'wb') as fout:
            fout.write(json.dumps(self.params, sort_keys=True))
        with so.smart_open(s3_keys['doc_ids'], 'wb') as fout:
            for i in document_ids:
                fout.write(i + '\n')

    def load_model(self, model_name):
        s3_keys = self._get_s3_keys(model_name)
        self.model = self.model.load(s3_keys['model'])

    def _get_s3_keys(self, model_name):
        return {
            'model': create_model_key(model_name, 'model', 'w2v'),
            'params': create_model_key(model_name, 'params', 'json'),
            'doc_ids': create_model_key(model_name, 'doc_ids', 'txt')
        }

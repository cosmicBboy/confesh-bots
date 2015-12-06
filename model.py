'''Module for running supervised models on confesh text data'''

import pandas as pd
import numpy as np
import random
import logging
import textmining
from sklearn import linear_model
from scipy import sparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def create_learning_set(target_array, feature_array, n=0.8):
    train_n = int(feature_array.shape[0] * n)
    index = range(feature_array.shape[0])
    random.shuffle(index)
    train_i = index[:train_n]
    test_i = index[train_n:]
    training_set = {
        'features': feature_array[:train_n],
        'targets': target_array[:train_n]
    }
    test_set = {
        'features': feature_array[train_n:],
        'targets': target_array[train_n:]
    }
    return {'training_set': training_set,
            'test_set': test_set}


class ModelPipeline(object):

    def __init__(self, model, target_col, feature_col, **params):
        self.target_col = target_col
        self.feature_col = feature_col
        self.params = params
        self.model = model(**self.params)

    def prepare_sparse_corpus(self, corpus_df):
        y, X = self.prepare_sparse_dtm(corpus_df,
                                       self.target_col,
                                       self.feature_col)
        self.target_array = y
        self.feature_array = X
        return self

    def prepare_learning_set(self):
        learning_set = create_learning_set(self.target_array,
                                           self.feature_array)
        self.training_set = learning_set['training_set']
        self.test_set = learning_set['test_set']
        logging.info("TRAIN SET:\n{}".format(self.training_set))
        logging.info("TEST SET:\n{}".format(self.test_set))
        return self

    def train_model(self):
        y = self.training_set['targets']
        X = self.training_set['features']
        self.model.fit(X, y)
        return self

    def predict(self):
        X_test = self.test_set['features']
        self.predictions = self.model.predict(X_test)
        return self

    def evaluate(self):
        y_test = self.test_set['targets']
        X_test = self.test_set['features']
        self.coeffs = self.generate_coeffs()
        self.mse = np.mean((self.predictions - y_test) ** 2)
        self.variance = self.model.score(X_test, y_test)
        return self

    def prepare_sparse_dtm(self, df, target_col, feature_col):
        y = np.ravel(list(df[target_col]))
        dfX = df[feature_col]
        dtmX = self.create_dtm(dfX)
        return y, dtmX

    def create_dtm(self, df):
        docs = [d[1].split() for d in df.iteritems()]
        self.indptr = [0]
        self.indices = []
        self.data = []
        vocabulary = {}
        for d in docs:
            for term in d:
                index = vocabulary.setdefault(term, len(vocabulary))
                self.indices.append(index)
                self.data.append(1)
            self.indptr.append(len(self.indices))
        self.vocabulary = {v: k for k, v in vocabulary.items()}
        token_matrix = sparse.csr_matrix(
            (self.data, self.indices, self.indptr), dtype=int
        )
        intercept = sparse.csr_matrix(
            np.array([[1] for i in range(df.shape[0])])
        )
        dtm = sparse.hstack([intercept, token_matrix], format='csr')
        return dtm

    def generate_coeffs(self):
        coef = self.model.coef_
        print self.vocabulary
        vocab_list = ['Intercept']
        vocab_list.extend([self.vocabulary[i] for i in range(len(coef) - 1)])
        co_df = pd.DataFrame(zip(coef, vocab_list), columns=['coeff', 'token'])
        # return co_df
        return co_df.sort_values('coeff', ascending=False)

    def summarize_model(self):
        logging.info("PREDICTIONS:\n{}".format(self.predictions))
        logging.info("COEFFICIENTS:\n{}".format(self.coeffs))
        logging.info("MEAN SQUARE ERROR: {}".format(self.mse))
        logging.info("VARIANCE SCORE: {}".format(self.variance))

if __name__ == "__main__":
    INPUT_FP = "./tmp/clean/holyokecon_confessional_secret_tokens.csv"
    data = pd.read_csv(INPUT_FP)
    data.dropna(how='any', inplace=True)

    PARAMS = {
        'alpha': 0.1,
        'normalize': True
    }
    TARGET_COLUMN = 'comments'
    FEATURE_COLUMN = 'stemmed_words'

    pipeline = ModelPipeline(linear_model.Ridge,
                             TARGET_COLUMN, FEATURE_COLUMN, **PARAMS)

    print pipeline.model
    pipeline.prepare_sparse_corpus(data)
    pipeline.prepare_learning_set()
    pipeline.train_model()
    pipeline.predict()
    pipeline.evaluate().summarize_model()




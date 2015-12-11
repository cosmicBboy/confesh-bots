#!/usr/bin/python
# -*- coding: utf-8 -*-

'''A module for training a topic model on a corpus'''
import logging
import pandas as pd
from gensim import corpora, models, similarities
from gensim.models.ldamodel import LdaModel
from preprocess import Preprocessor, CLEAN_COLUMN, TOKEN_COLUMN
from preprocess import tokenize_words
from preprocess import remove_word_len_thres


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class Pipeline(object):

    def __init__(self, preprocessor):
        self.ppr = preprocessor

    def run_w_token_pipeline(self):
        logging.info("Running word token pipeline")
        self.ppr.run_pipelines([
                tokenize_words(CLEAN_COLUMN, TOKEN_COLUMN, "Tokenizing words"),
                # remove_word_len_thres(CLEAN_COLUMN, None,
                #               "Removing words with 2 or less characters"),
            ])

    def run_model_pipeline(self, model_class, col_selector, model_params,
                           dict_fp='', corpus_fp='', model_fp=''):
        self._set_texts(col_selector)
        self._generate_dictionary(dict_fp)
        self._generate_corpus(corpus_fp)
        self._generate_tfidf_corpus()
        self._generate_model(model_class, model_fp, model_params)

    def run_model_pipelines(self, pipelines, model_class, output_fp):
        for p in pipelines:
            experiment_name = p['model_name']
            logging.info("Running model experiment: %s" % experiment_name)
            model_params = {k: v for k, v in p.items()
                            if k not in ['model_name']}
            logging.info("Model Parameters: %s" % model_params)
            dict_fp='%s/dicts/%s.dict' % (output_fp, experiment_name)
            corpus_fp='%s/corpus/%s.mm' % (output_fp, experiment_name)
            model_fp='%s/models/%s.lda' % (output_fp, experiment_name)
            self.run_model_pipeline(model_class, TOKEN_COLUMN,
                                    model_params, dict_fp=dict_fp, corpus_fp=corpus_fp,
                                    model_fp=model_fp)
            self.pretty_print_lda_model(num_topics=50)

    def pretty_print_lda_model(self, **kwargs):
        for _ in range(10):
            print("---------------------------")

        for topics in self.model.show_topics(kwargs):
            print topics.encode('ascii', 'ignore')
            print '\n'

    def _set_texts(self, col_selector):
        self.texts = self.ppr.df[col_selector].copy()

    def _generate_dictionary(self, fp):
        self.dictionary = corpora.Dictionary(self.texts)
        self.dictionary.save(fp)

    def _generate_corpus(self, fp):
        self.corpus = self.texts.apply(self.dictionary.doc2bow)
        corpora.MmCorpus.serialize(fp, self.corpus)

    def _generate_tfidf_corpus(self):
        tfidf = models.TfidfModel(self.corpus)
        self.corpus = tfidf[self.corpus]

    def _generate_model(self, model_class, fp, model_params,
                        target_col='model'):
        print model_params

        model = model_class(self.corpus,
                            id2word=self.dictionary,
                            **model_params)

        setattr(self, target_col, model)
        self.model.save(fp)


if __name__ == "__main__":

    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('-i', help='input file')
    p.add_argument('-o', help='output file')

    args = p.parse_args()
    input_fp = args.i
    output_fp = args.o

    data = pd.read_csv(input_fp)

    # Preparing Data
    d = data.copy()
    d[CLEAN_COLUMN] = d[CLEAN_COLUMN].astype(str)
    p = Preprocessor(d)

    pline = Pipeline(p)
    pline.run_w_token_pipeline()
    print pline.ppr.df.head()

    lda_model_specs_secrets_pdd1 = {
        'model_name': 'lda_model_specs_secrets_pdd13',
        'alpha': 'auto',
        'num_topics': 5,
        'passes': 5,
        'update_every': 1,
        'chunksize': 5000
    }

    lda_model_specs_secrets_pdd2 = {
        'model_name': 'lda_model_specs_secrets_pdd14',
        'alpha': 'auto',
        'num_topics': 10,
        'passes': 5,
        'update_every': 1,
        'chunksize': 5000
    }

    lda_model_specs_secrets_pdd3 = {
        'model_name': 'lda_model_specs_secrets_pdd15',
        'alpha': 'auto',
        'num_topics': 15,
        'passes': 5,
        'update_every': 1,
        'chunksize': 5000
    }

    lda_model_specs_secrets_pdd4 = {
        'model_name': 'lda_model_specs_secrets_pdd16',
        'alpha': 'auto',
        'num_topics': 20,
        'passes': 10,
        'update_every': 1,
        'chunksize': 5000
    }

    lda_model_specs_secrets_pdd5 = {
        'model_name': 'lda_model_specs_secrets_pdd17',
        'alpha': 'auto',
        'num_topics': 20,
        'passes': 10,
        'update_every': 1,
        'chunksize': 5000
    }

    model_pipelines = [
        lda_model_specs_secrets_pdd1,
        lda_model_specs_secrets_pdd2,
        lda_model_specs_secrets_pdd3,
        lda_model_specs_secrets_pdd4,
        lda_model_specs_secrets_pdd5
    ]

    pline.run_model_pipelines(model_pipelines, LdaModel, output_fp)
    pline.pretty_print_lda_model()

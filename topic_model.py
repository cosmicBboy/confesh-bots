#!/usr/bin/python
# -*- coding: utf-8 -*-

'''A module for training a topic model on a corpus'''
import logging
import pandas as pd
from gensim import corpora, models, similarities
from preprocess import Preprocessor

LdaModel = models.ldamodel.LdaModel

class Pipeline(object):

    def __init__(self, preprocessor):
        self.ppr = preprocessor

    def run_w_token_pipeline(self):
        logging.info("Running word token pipeline")
        self.ppr.run_pipelines([
                self.ppr.w_tokenize,
                p.rm_words_threshold,
                self.ppr.rm_punctuation,
                self.ppr.rm_stopwords,
                self.ppr.w_lemm,
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
            self.run_model_pipeline(model_class, "word_tokens",
                                    model_params, dict_fp=dict_fp, corpus_fp=corpus_fp,
                                    model_fp=model_fp)
            pline.pretty_print_lda_model(num_topics=50)

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


fp = "./tmp/holyokecon_confessional_secrets.csv"
data = pd.read_csv(fp)

# Preparing Data
d = data.copy()
p = Preprocessor(d)

pline = Pipeline(p)
pline.run_w_token_pipeline()
print pline.ppr.df


lda_model_specs_secrets_pdd1 = {
    'model_name': 'lda_model_specs_secrets_pdd9',
    'num_topics': 50,
    'passes': 5,
    'update_every': 1,
    'chunksize': 5000
}

lda_model_specs_secrets_pdd2 = {
    'model_name': 'lda_model_specs_secrets_pdd10',
    'num_topics': 25,
    'passes': 5,
    'update_every': 1,
    'chunksize': 5000
}

lda_model_specs_secrets_pdd3 = {
    'model_name': 'lda_model_specs_secrets_pdd11',
    'alpha': 'auto',
    'num_topics': 50,
    'passes': 5,
    'update_every': 1,
    'chunksize': 5000
}

lda_model_specs_secrets_pdd4 = {
    'model_name': 'lda_model_specs_secrets_pdd12',
    'num_topics': 50,
    'passes': 10,
    'update_every': 1,
    'chunksize': 5000
}

model_pipelines = [
    lda_model_specs_secrets_pdd1,
    lda_model_specs_secrets_pdd2,
    lda_model_specs_secrets_pdd3,
    lda_model_specs_secrets_pdd4
]

output_fp = "./topic_modelling"
pline.run_model_pipelines(model_pipelines, LdaModel, output_fp)
pline.pretty_print_lda_model(num_topics=50)

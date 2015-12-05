# !/usr/bin/python
# -*- coding: utf-8 -*-

'''A module for cleaning, tokenizing and feature selection'''
import logging
import pandas as pd
import unicodedata as utfd
import re
import string
import nltk
import textmining
from argparse import ArgumentParser
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def pipeline_decorator(func):
    def pipeline_wrapper(column, target, message):
        return {'func': func,
                'column': column,
                'target': target,
                'message': message}
    return pipeline_wrapper

@pipeline_decorator
def to_ascii(text):
    return text.decode('utf-8')

@pipeline_decorator
def lowercase(text):
    return text.lower()

@pipeline_decorator
def remove_punctuation(text):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub("", text)

@pipeline_decorator
def remove_numbers(text):
    regex = re.compile('[0-9]')
    return regex.sub("", text)

@pipeline_decorator
def tokenize_words(text):
    return word_tokenize(text)

@pipeline_decorator
def remove_word_len_thres(tokens, length=2):
    return [w for w in tokens if len(w) > length]

@pipeline_decorator
def remove_stopwords(tokens):
    return [w for w in tokens if w not in stop_words]

@pipeline_decorator
def remove_stopwords_in_sents(sent_token_list):
    return [remove_stopwords(s) for s in sent_token_list]

@pipeline_decorator
def stem_words(tokens):
    return [ps.stem(w) for w in tokens]

@pipeline_decorator
def lemmatize_words(tokens):
    return [lemmatizer.lemmatize(w) for w in tokens]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class Preprocessor(object):

    def __init__(self, dataframe):
        self.df = dataframe
        # self.to_ascii().to_lowercase().rm_punctuation().rm_numbers()

    def map(self, func, col_name, target_col_name=None):
        '''
        A mapping method that applies func to a col_name series.
        If target_col_name is specified, then the result of the apply
        is assigned to a new column. By default assigns new values to
        col_name
        '''
        if not target_col_name:
            self.df[col_name] = self.df[col_name].apply(func)
            print self.df[col_name].head()
        else:
            try:
                self.df[target_col_name] = self.df[col_name].apply(func)
                print self.df[target_col_name].head()
            except ValueError as e:
                logging.error(e)

    def run_pipelines(self, *pipelines):
        for func_list in pipelines:
            for func_dict in func_list:
                logging.info(func_dict['message'])
                print func_dict['func']
                self.map(func_dict['func'],
                         func_dict['column'],
                         func_dict['target'])
                print ""
        return self

    def create_tdm(self, text_var):
        '''
        Restructures self.df to contain the following columns:
        - Unique Identifier
        - Outcome Variable
        - Text Features
        '''
        tdm = textmining.TermDocumentMatrix()
        for doc in self.df[text_var].apply(lambda x: " ".join(x)).iteritems():
            tdm.add_doc(doc[1])
        return tdm

    def to_csv(self, output_fp, id_key, outcome_col, text_col, **kwargs):
        self.df[text_col] = self.df[text_col].apply(lambda x: " ".join(x))
        csv_df = self.df[[id_key, outcome_col, text_col]].copy()
        csv_df.to_csv(output_fp, **kwargs)


if __name__ == "__main__":
    parser = ArgumentParser(description='Proprocessing Confesh text data')
    parser.add_argument('-i', help='Input filepath')
    parser.add_argument('-o', help='Output filepath')
    parser.add_argument('-nrows', type=int, default=None,
                        help='number of rows to read in')
    args = parser.parse_args()
    output_fp = args.o
    d = pd.read_csv(args.i, nrows=args.nrows)

    ID_KEY = "id"
    RAW_COLUMN = 'confession'
    TOKEN_COLUMN = 'word_tokens'
    CLEAN_COLUMN = "stemmed_words"
    OUTCOME_COLUMN = 'comments'

    p = Preprocessor(d)

    preprocess_pipeline = [
        to_ascii(RAW_COLUMN, None, "Encoding to ascii"),
        lowercase(RAW_COLUMN, None, "Making text lowercase"),
        remove_punctuation(RAW_COLUMN, None, "Removing Punctuation"),
        remove_numbers(RAW_COLUMN, None, "Removing Numbers"),
    ]

    token_pipeline = [
        tokenize_words(RAW_COLUMN, TOKEN_COLUMN, "Tokenizing words"),
        remove_word_len_thres(TOKEN_COLUMN, None,
                              "Removing words with 2 or less characters"),
        remove_stopwords(TOKEN_COLUMN, None, "Removing stopwords from text"),
        lemmatize_words(TOKEN_COLUMN, CLEAN_COLUMN,
                        "Lemmatizing words in text"),
    ]

    p.run_pipelines(
        preprocess_pipeline,
        token_pipeline,
    )

    p.to_csv(output_fp, ID_KEY, OUTCOME_COLUMN, CLEAN_COLUMN,
             index=False, encoding='utf-8')

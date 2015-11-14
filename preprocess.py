 #!/usr/bin/python
# -*- coding: utf-8 -*-

'''A module for cleaning, tokenizing and feature selection'''
import logging
import pandas as pd
import unicodedata as utfd
import re
import string
import nltk
from argparse import ArgumentParser
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def to_ascii(text):
    return text.decode('utf-8')

def lowercase(text):
    return text.lower()

def remove_punctuation(text):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub("", text)

def remove_numbers(text):
    regex = re.compile('[0-9]')
    return regex.sub("", text)

def remove_word_len_thres(tokens, length=2):
    return [w for w in tokens if len(w) > length]

def remove_stopwords(tokens):
    return [w for w in tokens if w not in stop_words]

def remove_stopwords_in_sents(sent_token_list):
    return [remove_stopwords(s) for s in sent_token_list]

def stem_words(tokens):
    return [ps.stem(w) for w in tokens]

def lemmatize_words(tokens):
    return [lemmatizer.lemmatize(w) for w in tokens]

def lemmatize_words_in_sents(tokens):
    return [lema(w) for w in tokens]

def stem_words_in_sents(sent_token_list):
    return [lemmatize_words(s) for s in sent_token_list]

def word_tokenize_sents(sent_list):
    return [word_tokenize(s) for s in sent_list]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class Preprocessor(object):

    def __init__(self, dataframe):
        self.df = dataframe
        self.to_ascii().to_lowercase().rm_punctuation().rm_numbers()

    def map(self, func, col_name, target_col_name='', inplace=False):
        if inplace:
            self.df[col_name] = self.df[col_name].apply(func)
        else:
            try:
                self.df[target_col_name] = self.df[col_name].apply(func)
            except ValueError as e:
                logging.error(e)

    def log_info(self, message):
        logging.info("%s" % (message))

    def to_ascii(self):
        self.log_info("Encoding to ascii")
        self.map(to_ascii, 'confession', inplace=True)
        return self

    def rm_punctuation(self):
        self.log_info("Removing Punctuation")
        self.map(remove_punctuation, 'confession', inplace=True)
        return self

    def rm_numbers(self):
        self.log_info("Removing Punctuation")
        self.map(remove_numbers, 'confession', inplace=True)
        return self

    def to_lowercase(self):
        self.log_info("Making text lowercase")
        self.map(lowercase, 'confession', inplace=True)
        return self

    def rm_words_threshold(self):
        self.log_info("Removing words with length character of 2 or less")
        self.map(remove_word_len_thres, 'word_tokens', inplace=True)

    def rm_stopwords(self):
        self.log_info("Removing stopwords from text")
        self.map(remove_stopwords, 'word_tokens', inplace=True)
        return self

    def rm_stopwords_sents(self):
        self.log_info("Removing stopwords sentences")
        self.map(remove_stopwords_in_sents, 'word_sent_tokens', inplace=True)
        return self

    def s_tokenize(self):
        self.log_info("Tokenizing sentences")
        self.map(sent_tokenize, 'confession', 'sent_tokens')
        return self

    def s_count(self):
        self.log_info("Counting sentences")
        self.map(len, 'sent_tokens', 'num_sents')
        return self

    def w_tokenize(self):
        self.log_info("Tokenizing words")
        self.map(word_tokenize, 'confession', 'word_tokens')
        return self

    def w_tokenize_s(self):
        self.log_info("Tokenizing words in sentences")
        self.map(word_tokenize_sents, 'sent_tokens', 'word_sent_tokens')
        return self

    def w_stem(self):
        self.log_info("Stemming words in text")
        self.map(stem_words, 'word_tokens', 'stemmed_words')
        return self

    def w_stem_s(self):
        self.log_info("Stemming words in sentences")
        self.map(stem_words_in_sents, 'word_sent_tokens', 'stemmed_sents')
        return self

    def w_lemm(self):
        self.log_info("Lemmatizing words in text")
        self.map(lemmatize_words, 'word_tokens', 'stemmed_words')
        return self

    def w_lemm_s(self):
        self.log_info("Lemmatizing words in sentences")
        self.map(lemmatize_words_in_sents, 'word_sent_tokens', 'stemmed_sents')
        return self

    def w_pos_tag(self):
        self.log_info("Part-of-speech tagging text")
        self.map(nltk.pos_tag, 'word_tokens', 'pos_tag')
        return self

    def run_pipelines(self, *pipelines):
        for func_list in pipelines:
            for func in func_list:
                func()


if __name__ == "__main__":
    fp = "./tmp/holyokecon_confessional_secrets.csv"
    data = pd.read_csv(fp)

    d = data.copy()
    d = d.head(5)

    p = Preprocessor(d)

    get_w_token_pipeline = [
        p.w_tokenize,
        p.rm_words_threshold,
        p.rm_punctuation,
        p.rm_stopwords,
        p.w_lemm,
    ]

    get_ws_token_pipeline = [
        p.s_tokenize,
        p.s_count,
        p.w_tokenize_s,
        p.rm_stopwords_sents,
        p.w_stem_s,
    ]

    p.run_pipelines(
        get_w_token_pipeline,
        # get_ws_token_pipeline
    )

    print p.df['word_tokens']

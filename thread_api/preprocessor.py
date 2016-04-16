'''Module that Prepares Confesh Secrets and Comments for Modeling
'''

import mongo_creds as creds
import re
import string
from stream_mongo import MongoStreamer
from argparse import ArgumentParser
from funcy import compose
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

confesh_stopwords = [
    'http',
    'com'
]
stop_words = set(stopwords.words('english') + confesh_stopwords)


class TextPreprocessor():

    def __init__(self):
        pass

    def preprocess(self, text):
        '''Wraps all preprocessing functions
        '''
        return compose(self._tokenize,
                       self._remove_punctuation,
                       self._remove_numbers,
                       self._lowercase)(text)

    def _lowercase(self, text):
        return text.lower()

    def _remove_numbers(self, text):
        return re.compile('[0-9]').sub(" ", text.lower())

    def _remove_punctuation(self, text):
        return re.compile('[%s\.]' % re.escape(string.punctuation)).sub(
            "", text)

    def _tokenize(self, text, len_thres_lt=1, len_thres_gt=20):
        return [w for w in word_tokenize(text)
                if len(w) > len_thres_lt and
                len(w) < len_thres_gt and
                w not in stop_words]

'''
Module for preprocessing dream corpus into interpretations

For each entry in a dream corpus:

vocab1: definition1
vocab2: definition2
vocab3: definition3
...

We want to split the definitions by interpretation. We define an interpretation
as a nuanced definition of a dream entry.

Example: Abbey

Interpretation 1:
To see an abbey in your dream signifies spirituality, peace of mind and freedom
from anxiety. You are in a state of contentment and satisfaction. Help for you
is always around the corner.

Interpretation 2:
To see an abbey in ruins indicates feelings of hopelessness. You have a
tendency to not finish what you started
'''

import pandas as pd
import numpy as np
import re
import logging
from nltk.tokenize import sent_tokenize

from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

STOP_PATTERNS =[
    'please see',
    'also',
    'dream moods\' interpretation for',
    'common dream themes:'
]
VOCAB_DELIMITER = ';'

def parse_definition(definition):
    sents = sent_tokenize(definition)
    return sents


def find_vocab(text, delim='or'):
    # heuristic for finding vocab words: split by '.' and use string
    # in the 0th index
    text = text.split('.')[0]
    text = re.sub('<end>', '', text)
    for p in STOP_PATTERNS:
        text = re.sub(p, '', text)
    return VOCAB_DELIMITER.join(text.split(' {} '.format(delim)))


def preprocess_dreams(dream_df):
    dream_df.loc[ : , 'definitions'] = \
        dream_df['definitions'].apply(lambda x: "{}{}".format(x, "<end>"))

    # heuristic for extracting referral phrases
    refer_df = dream_df[dream_df['definitions'].str.contains("please see")]
    refer_extract = refer_df['definitions'].str.extract("(please see.+?<end>)")

    # find vocab words
    refer_vocab = refer_extract.apply(find_vocab)

    # add column to dream_df
    dream_df['redirect'] = pd.Series([np.nan for _ in range(dream_df.shape[0])])
    dream_df.loc[refer_vocab.index, 'redirect'] = refer_vocab
    dream_df.loc[ : , 'definitions'] = \
        dream_df['definitions'].apply(lambda x: re.sub('<end>', '', x))
    return dream_df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', help='clean dreams filepath', type=str)
    parser.add_argument('-o', help='complete dreams filepath', type=str)
    args = parser.parse_args()

    dream_df = pd.read_csv(args.i)
    preprocess_dreams(dream_df).to_csv(args.o, index=False, encoding='utf-8')

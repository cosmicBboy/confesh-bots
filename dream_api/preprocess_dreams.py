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

STOP_PATTERNS = [
    'please see',
    'also',
    'dream moods\' interpretation for',
    'common dream themes:'
]
INTERP_DELIM = [
    'to',
    'if',
    'alternatively',
    'in particular',
    'that',
]

VOCAB_DELIMITER = ';'


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


def interp_aggregator(dream_group):
    '''Aggregator function for dream corpus groups
    '''
    interps = dream_group['definitions'].apply(
        parse_interpretations).tolist()[0]
    redirect = [dream_group['redirect'].iloc[0] for _ in interps]
    interp_df = pd.DataFrame({'interpretations': interps,
                              'redirect': redirect})
    return interp_df


def parse_interpretations(dream_entry):
    '''Parses a single dream entry into interpretations

    Input: dream_entry (String)
    Output: a pandas Series of dream interpretations (Series of Strings)
    '''

    # TODO: write a test that checks whether the empty dream interpretations
    # are the same rows what look like "*please see <other entry>"

    sents = sent_tokenize(dream_entry)
    interp_index = [(i, s) for (i, s) in enumerate(sents)
                    if contains_interp_delimiter(s)]
    index = [i[0] for i in interp_index]
    if len(index) <= 1:
        # return original sentence if no interpretations are detected
        # or if there is only one interpretation
        return [dream_entry]

    if index[0] != 0:
        index = [0] + index
    interp = format_interpretation(sents, zip(index[:-1], index[1:]))
    return interp


def contains_interp_delimiter(sentence):
    '''Check if dream entry sentence contains interpretation delimiter
    '''
    delims = ["^{}".format(d) for d in INTERP_DELIM]
    return any([re.search(p, sentence) for p in delims])


def format_interpretation(sentences, indices):
    '''Formats a list of dream sentences into interpretations

    Uses indices to determine which sentences to group together
    '''
    return [" ".join(sentences[i[0]: i[1]]) for i in indices]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', help='clean dreams filepath', type=str)
    parser.add_argument('-o', help='complete dreams filepath', type=str)
    args = parser.parse_args()

    dream_df = pd.read_csv(args.i)
    dream_corpus = preprocess_dreams(dream_df)
    dream_interp = dream_corpus.groupby(['vocab']).apply(interp_aggregator)\
        .reset_index(level=0)
    dream_interp.to_csv(args.o, index=False, encoding='utf-8')
    print dream_interp.head(50)

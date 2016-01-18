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
from nltk.tokenize import sent_tokenize


def parse_definition(definition):
    sents = sent_tokenize(definition)
    return sents


if __name__ == "__main__":
    dream_df = pd.read_csv('../data/dream_corpus.csv')
    print dream_df.head(25)

    # REFER_PATTERN = '\*please'
    # refer_df = dream_df['definitions'].str.contains(REFER_PATTERN)
    # print refer_df

    # definition = dream_df['definitions'][dream_df['vocab'] == 'abbey'].tolist()
    # interpretations = parse_definition(definition[0])
    # print interpretations

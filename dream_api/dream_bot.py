'''
Dream Bot

This bot parses secrets from dream.confesh.com and replies to the secret with
a top sentence to interpret the dream based on this process:

Obtain Secrets from dreams.confesh.com
1.1 Connect to Mongo database, create cursor stream for secret.
1.2 Read in complete dream corpus.

Train a Word2Vec Model
2.1 Concatenate dream secrets with dream corpus definition entries.
2.2 Train a Word2Ven model using Gensim

Matching Dream Interpretations with Secrets
3.1 tokenize the secret (unigrams and bigrams), remove stopwords, filter freq.
3.2 For each secret token, find dream corpus entries that contain the token.
3.3 For each match, get the dream definition and sentence tokenize the
    definition string.
3.4 Take the original tokenized secret and compute rank-ordered similarity of
    all matching sentences using Word2Vec model.
3.5 Select top ranked sentence and insert it to the top of the comments field
    of the corresponding secret MongoDB document.
'''


def read_mongo(connection_string):
    pass

def read_dream_corpus():
    pass

def prep_data():
    pass

def train_model():
    pass

def load_model():
    pass

def parse_secret():
    pass

def match_tokens_to_dreams():
    pass

def prep_dream_defitions():
    pass

def rank_interpretations():
    pass

def insert_interpretation():
    pass

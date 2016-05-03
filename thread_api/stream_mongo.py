'''Module for Ingesting Secrets and Comments from Confesh

The cursor schema is as follows:

    '_id': unique id
    'text': secret text
    'comments': list of comments
    'tags': list of tags
    'bumped': post has been bumped
'''

import json
import mongo_creds as creds
from argparse import ArgumentParser
from pymongo import MongoClient
from bson.objectid import ObjectId
from itertools import chain

TBOT = 'threadbot'


def _encode_utf(text):
    return unicode(text).encode('utf-8')


class MongoStreamer(object):

    def __init__(self, domain, port, db, coll):
        self.domain = domain
        self.port = port
        self.db = db
        self.coll = coll
        self.client = MongoClient(self.domain, self.port)[self.db][self.coll]

    def iterate_secrets_comments(self, community_name, limit=0):
        cursor = self._read_secrets(community_name, limit)
        return chain(*(self._secret_and_comment_obj_handler(doc)
                     for doc in cursor))

    def iterate_secrets(self, community_name, limit=0):
        cursor = self._read_secrets(community_name, limit)
        return chain(*(self._secret_obj_handler(doc) for doc in cursor))

    def _read_secrets(self, community_name, limit):
        query = {'communities': community_name}
        projection = {'_id': 1, 'text': 1, 'comments': 1,
                      'tags': 1, 'bumped': 1, 'views': 1,
                      'status': 1}
        return self.client.find(query, projection, limit=limit)

    def _secret_and_comment_obj_handler(self, secret_obj):
        '''Flattens secret and comment objects from cursor

        input: secret_obj (dict)
        output: iterable of secret and comments (generator)
        '''
        get_threadbot_bool = False
        result = [self._doc2dict(secret_obj, 'SECRET', get_threadbot_bool)]
        if secret_obj.get('comments', None):
            comments = [self._doc2dict(c, 'COMMENT', get_threadbot_bool)
                        for c in secret_obj['comments']]
            result = result + comments
        return result

    def _secret_obj_handler(self, secret_obj):
        '''Flattens secret object from cursor object

        input: secret_obj (dict)
        output: iterable of secrets (generator)
        '''
        result = self._doc2dict(secret_obj, 'SECRET', True)
        return [result]

    def _doc2dict(self, secret_obj, doc_type, get_threadbot_bool):
        '''Converts secret object with mongo schema to threadbot schema

        Input:
        secret_obj (dict): a dictionary with the secret schema in the mongo
                           database as defined by the projection in
                           _read_secrets
        doc_type (dict): 'SECRET' or 'COMMENT'
        get_threadbot_bool (bool): if True, checks if there are any comments
                                   in a secret made by the threadbot
        '''
        if doc_type not in ['SECRET', 'COMMENT']:
            msg = 'Unrecognized doc_type: {}. Must be {}'.format(
                doc_type, )
            raise ValueError(msg)

        result = {'id': str(secret_obj['_id']),
                  'text': _encode_utf(secret_obj['text']),
                  'bumped': secret_obj.get('bumped', False),
                  'doc_type': doc_type}

        if doc_type == 'SECRET' and get_threadbot_bool:
            result['views'] = secret_obj.get('views', 0)
            result['hidden'] = secret_obj.get('status', None) == 'HIDDEN'
            result['contains_threadbot_post'] = \
                self._secret_contains_threadbot_post(secret_obj)
        return result

    def _secret_contains_threadbot_post(self, secret_obj):
        '''Returns true if secret contains threadbot comment
        '''
        comments = secret_obj.get('comments', None)
        if comments:
            tbot_comments = [c for c in comments if c.get('avatar', None)
                             and c.get('avatar').get('text', None)]
            contains_threadbot_post = any([c for c in tbot_comments
                                           if c['avatar']['text'] == TBOT])
            return contains_threadbot_post
        else:
            return False

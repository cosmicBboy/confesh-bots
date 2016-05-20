import json
from argparse import ArgumentParser
from pymongo import MongoClient
from mongo_creds import domain, port
from bson.objectid import ObjectId

FIELD_LIST = ['_id', 'text', 'comments', 'avatar']

def create_confesh_stream(db_name, collection_name, query, **kwargs):
    coll_obj = fetch_collection(db_name, collection_name)
    query_results = collection_find(coll_obj, query)
    return text_stream_cursor(query_results, FIELD_LIST, **kwargs)

def fetch_collection(db_name, collection_name):
    '''Gets collection mongo object'''
    client = MongoClient(domain, port)
    return client[db_name][collection_name]

def collection_find(collection_obj, query):
    '''Returns results of a query on a collection'''
    return collection_obj.find(query)

def text_stream_cursor(cursor, field_list, n=5):
    '''Streams the text and unique ids for query results'''
    if n:
        cursor = cursor[:n]
    for result in cursor:
        yield {k: v for k, v in result.items()
               if k in field_list}

if __name__ == "__main__":
    from datetime import datetime
    DATETIME_THRES = datetime(2016, 3, 20, 0, 00, 00, 000000)
    # Example queries:
    # query = {'communities': 'bots'}
    # query = {'_id': ObjectId('56ef6601e4b07260f818b904')}

    query_id = '573e8d9be4b07260f81a4fb9'
    query = {'communities': 'bots', '_id': ObjectId(query_id)}
    confesh_stream = create_confesh_stream('confesh-db', 'confession', query,
                                           n=None)

    # This removes comments that meet the conditional filters
    result_comment_ids = []
    for result in confesh_stream:
        for k, v in result.items():
            print k
            print v
            print ''
    #     comments = result.get('comments', None)
    #     if comments:
    #         # if there are comments in the post, then collect dream._ids
    #         result_comments = [c for c in comments if
    #                            c.get('avatar', None)]
    #         ids = [c['_id'] for c in result_comments if
    #                c.get('avatar')['text'] == 'HelpulDrake' and
    #                c.get('timestamp') > DATETIME_THRES]
    #         result_comment_ids.extend(ids)

    # # TODO: Functionalize this section
    # coll_obj = fetch_collection('confesh-db', 'confession')
    # coll_obj.update_one({'_id': ObjectId(query_id)},
    #                     {'$pull': {'comments': {'_id': {"$in": result_comment_ids}}}})

    # # Now need to update number of comments field based on length of comments array
    # coll_obj = fetch_collection('confesh-db', 'confession')
    # for result in coll_obj.find(query):
    #     secret_id = str(result['_id'])
    #     comments = result.get('comments', None)
    #     if comments:
    #         len_comments = len(comments)
    #         coll_obj.update_one(
    #             {'_id': ObjectId(secret_id)}, {'$set': {'numberOfComments': len_comments}})
    #     else:
    #         coll_obj.update_one(
    #             {'_id': ObjectId(secret_id)}, {'$set': {'numberOfComments': 0}})

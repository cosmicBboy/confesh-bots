import json
from argparse import ArgumentParser
from pymongo import MongoClient
from mongo_creds import domain, port

FIELD_LIST = ['_id', 'text']

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
    confesh_stream = create_confesh_stream('confesh-db', 'confession',
                                           {'communities': 'dreams'})
    for result in confesh_stream:
        print result

import json
import uuid
from collections import OrderedDict
from datetime import datetime
from bson.objectid import ObjectId
from query import create_confesh_stream, fetch_collection
from pymongo import MongoClient
from mongo_creds import domain, port

mongo_comment_schema = OrderedDict({
    u'text': None,
    u'_id': None,
    u'timestamp': None,
})


if __name__ == "__main__":
    confesh_stream = create_confesh_stream('confesh-db', 'confession',
                                           {'communities': 'bots'})
    insert_collection_conn = fetch_collection('confesh-db', 'confession')

    for result in confesh_stream:
        print result
    print '\n'

    comment = "test comment"
    post_id = ObjectId('56c3fa3fe4b01c56828d8f7c')
    bot_comment = mongo_comment_schema.copy()
    bot_comment['text'] = 'The third bot comment ever!'
    bot_comment['_id'] = str(uuid.uuid4())
    bot_comment['timestamp'] = datetime.now()

    print bot_comment
    query_filter = {'_id': post_id}
    update_ops = {'$push': {'comments': bot_comment}}
    insert_collection_conn.update_one(query_filter, update_ops)

    comments = insert_collection_conn.find_one({'_id': post_id})['comments']


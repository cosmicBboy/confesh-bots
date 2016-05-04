'''Helper functions for storing data in S3

Make sure that your ~/.aws directory is properly configured with your
aws_access_key_id and aws_secret_access_key
'''

import logging
import sys
import json
import os
import boto
from time import sleep
from boto.s3.key import Key
from bitly_utils import shorten_secret_url

S3 = 's3://'
BUCKET = 'bot-services'
BOT_KEY = 'threadbot'
BITLY_BUCKET = 'bitly-cache'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)


def create_model_key(model_name, metadata_key_name, metadata_ext):
    '''Creates an s3 filepath
    Input: str
    Output: str
    '''
    path = os.path.join(S3, BUCKET, BOT_KEY, model_name, metadata_key_name)
    return path + '.{}'.format(metadata_ext)


def model_exists(model_name, metadata_key_name='model', metadata_ext='w2v'):
    bucket = boto.connect_s3().get_bucket(BUCKET)
    key_string = "{}/{}/{}.{}".format(
        BOT_KEY, model_name, metadata_key_name, metadata_ext)
    key = Key(bucket, key_string)
    return key.exists()


class BitlyS3Cacher(object):
    '''A class for handling bitly HTTP get response caching

    Currently we are using s3 to cache results. The s3 key format is:
    s3://bitly-cache/<secret_id>

    And the contents of that key is a json file that contains metadata about
    the bitly link for the specified secret_id
    '''

    def __init__(self):
        self.bucket = boto.connect_s3().get_bucket(BITLY_BUCKET)

    def fetch_bitly_data(self, secret_id, community):
        '''If s3 key exists, returns a dictionary, otherwise send HTTP get request

        The response of the HTTP get request is then cached with the
        appropriate secret_id_key and returned.
        '''
        self.key = Key(self.bucket, secret_id)
        if self.key.exists():
            logging.info('{} exists, fetching data'.format(self.key))
            contents = self.key.get_contents_as_string()
            return json.loads(contents)
        else:
            logging.info("{} doesn't exist, sending bitly GET request"
                         .format(self.key))
            response = shorten_secret_url(secret_id, community)
            sleep(1)
            self._cache_bitly_as_json(response)
            return response

    def _cache_bitly_as_json(self, response):
        self.key.set_contents_from_string(json.dumps(response))

if __name__ == "__main__":
    secret_id = '5713db57e4b07260f8197fb9'
    bitly_cacher = BitlyS3Cacher()
    bitly_data = bitly_cacher.fetch_bitly_data(secret_id)
    print bitly_data['url']

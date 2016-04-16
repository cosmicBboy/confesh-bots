'''Helper functions for storing data in S3
'''

import os

S3 = 's3://'
BUCKET = 'bot-services'
BOT_KEY = 'threadbot'


def create_model_key(model_name, metadata_key_name, metadata_ext):
    '''Creates an s3 filepath
    Input: str
    Output: str
    '''
    path = os.path.join(S3, BUCKET, BOT_KEY, model_name, metadata_key_name)
    return path + '.{}'.format(metadata_ext)

'''Unit test suite for bitly integration
'''
import json
from nose.tools import assert_equals
from thread_api import bitly_utils


class TestBitlyUtils:
    def test_shorten_secret_url(self):
        '''shorten_secret_url should return <200> HTTP response where new_hash is 0
        '''
        # the hash for this secret_id should exist
        secret_id = '5713db57e4b07260f8197fb9'
        response = bitly_utils.shorten_secret_url(secret_id)
        assert_equals(response['status_code'], 200)
        assert_equals(response['data']['new_hash'], 0)

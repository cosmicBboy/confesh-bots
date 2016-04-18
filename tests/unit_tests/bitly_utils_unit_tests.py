'''Unit test suite for bitly integration
'''
from nose import tools
from mock import patch, MagicMock
from thread_api import bitly_utils


long_url = "https://some/long/url"
target_uri = 'https://api-ssl.bitly.com'
confesh_url = 'http://www.confesh.com'
confesh_path = '#/confessions/latest-activity/confession'


def fake_response_to_dict(response):
    return {"fake": "json_data"}


class FakeResponse():
    def __init__(self):
        self.text = '{"fake": "json_data"}'


class TestBitlyUtils:
    def test_create_request_payload_returns_correct_keys(self):
        '''_create_request_payload should return a dictionary with correct keys
        '''
        payload = bitly_utils._create_request_payload(long_url)
        tools.assert_equals('access_token' in payload, True)
        tools.assert_equals('longUrl' in payload, True)
        tools.assert_equals('domain' in payload, True)
        tools.assert_equals('response_format' in payload, True)

    def test_create_request_payload_returns_correct_values(self):
        '''_create_request_payload should return a dictionary with correct
        values
        '''
        payload = bitly_utils._create_request_payload(long_url)
        tools.assert_is_not_none(payload.get('access_token', None))
        tools.assert_is_not_none(payload.get('longUrl', None))
        tools.assert_is_not_none(payload.get('domain', None))
        tools.assert_is_not_none(payload.get('response_format', None))

    def test_create_request_payload_returns_correct_value_types(self):
        '''_create_request_payload should return a dictionary with correct
        values and correct types
        '''
        payload = bitly_utils._create_request_payload(long_url)
        tools.assert_is_instance(payload.get('access_token', None), str)
        tools.assert_is_instance(payload.get('longUrl', None), str)
        tools.assert_is_instance(payload.get('domain', None), str)
        tools.assert_is_instance(payload.get('response_format', None), str)

    def test_format_bitly_endpoint_contains_target_uri(self):
        '''_format_bitly_endpoint should return a string containing api-ssl
        target and the specified api_endpoint
        '''
        api_endpoint = 'some/endpoint'
        bitly_endpoint = bitly_utils._format_bitly_endpoint(api_endpoint)
        tools.assert_in(target_uri, bitly_endpoint)
        tools.assert_in(api_endpoint, bitly_endpoint)

    def test_format_confesh_long_url_contains_confesh_domain(self):
        '''_format_confesh_long_url should return a string containing confesh
        domain, confession path, and specified secret_id
        '''
        secret_id = 'abcdefghijklmnop'
        fake_url = bitly_utils._format_confesh_secret_long_url(secret_id)
        tools.assert_in(confesh_url, fake_url)
        tools.assert_in(confesh_path, fake_url)
        tools.assert_in(secret_id, fake_url)

    def test_format_confesh_data(self):
        '''_format_confesh_data should return only keys 'long_url' and 'url'
        '''
        fake_data = {
            'status_code': 200,
            'data': {
                'url': 'fake_short_url',
                'hash': 'fake_hash',
                'global_hash': 'fake_global_hash',
                'long_url': 'fake_long_url',
                'new_hash': 0
            },
            'status_txt': 'OK'
        }
        result = bitly_utils._format_bitly_response(fake_data)
        expected_keys = set(['long_url', 'url', 'new_hash', 'status_code'])
        check = expected_keys == set(result.keys())
        tools.assert_equals(check, True)

    def test_response_to_dict(self):
        '''_response_to_dict should return a dictionary
        '''
        fake_response = FakeResponse()
        result = bitly_utils._response_to_dict(fake_response)
        tools.assert_is_instance(result, dict)

    @patch('thread_api.bitly_utils._format_confesh_secret_long_url')
    @patch('thread_api.bitly_utils._response_to_dict')
    def test_shorten_secret_url_calls_format_confesh_secret_long_url(
            self, response_dict_mock, long_url_mock):
        '''shorten_secret_url should call _format_confesh_secret_long_url
        '''
        response_dict_mock.side_effect = fake_response_to_dict
        with patch('thread_api.bitly_utils.r.get') as get_mock:
            get_mock.return_value = FakeResponse()
            bitly_utils.shorten_secret_url('some_secret_id')
            tools.assert_true(long_url_mock.called)

    @patch('thread_api.bitly_utils._format_bitly_endpoint')
    @patch('thread_api.bitly_utils._response_to_dict')
    def test_shorten_secret_url_calls_format_bitly_endpoint(
            self, response_dict_mock, bitly_mock):
        '''shorten_secret_url should call _format_bitly_endpoint
        '''
        response_dict_mock.side_effect = fake_response_to_dict
        with patch('thread_api.bitly_utils.r.get') as get_mock:
            get_mock.return_value = FakeResponse()
            bitly_utils.shorten_secret_url('some_secret_id')
            tools.assert_true(bitly_mock.called)

    @patch('thread_api.bitly_utils._create_request_payload')
    @patch('thread_api.bitly_utils._response_to_dict')
    def test_shorten_secret_url_calls_create_request_payload(
            self, response_dict_mock, payload_mock):
        '''shorten_secret_url should call _create_request_payload
        '''
        response_dict_mock.side_effect = fake_response_to_dict
        with patch('thread_api.bitly_utils.r.get') as get_mock:
            get_mock.return_value = FakeResponse()
            bitly_utils.shorten_secret_url('some_secret_id')
            tools.assert_true(payload_mock.called)

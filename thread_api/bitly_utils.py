import requests as r
import json
from time import sleep
from collections import OrderedDict
from bitly_creds import access_token


def shorten_secret_url(secret_id, community):
    long_url = _format_confesh_secret_long_url(secret_id, community)
    endpoint = _format_bitly_endpoint('v3/shorten')
    payload = _create_request_payload(long_url)
    response = r.get(endpoint, params=payload)
    response = _response_to_dict(r.get(endpoint, params=payload))
    return _format_bitly_response(response)


def _response_to_dict(response):
    return json.loads(response.text)


def _create_request_payload(long_url, domain='bit.ly',
                            response_format='json'):
    return {
        'access_token': access_token,
        'longUrl': long_url,
        'domain': domain,
        'response_format': response_format
    }


def _format_bitly_endpoint(api_endpoint):
    target_uri = 'https://api-ssl.bitly.com'
    return '%s/%s' % (target_uri, api_endpoint)


def _format_confesh_secret_long_url(secret_id, community):
    confesh_url = 'http://{}.confesh.com'.format(community)
    confesh_path = '#/confessions/latest-activity/confession'
    return '%s/%s/%s' % (confesh_url, confesh_path, secret_id)


def _format_bitly_response(response_dict):
    d = {k: v for k, v in response_dict['data'].items()
         if k in ['long_url', 'url', 'new_hash']}
    d['status_code'] = response_dict['status_code']
    return d


if __name__ == "__main__":
    print shorten_secret_url('5713db57e4b07260f8197fb9')

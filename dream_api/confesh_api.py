import dream_auth_creds as creds
import json
import urllib, urllib2
import requests


def fetch_auth_token(community_name):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "action": "auth",
        "email": creds.passphrase
    }
    r = requests.post(creds.auth_url.format(community_name),
                      headers=headers, data=json.dumps(payload))
    return json.loads(r.content)['token']


def post_comment(community_name, secret_id, auth_token, text):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "authToken": auth_token
    }
    payload = {
        'confessionId': secret_id,
        'comment': {
            'text': text
        }
    }
    print "Payload:\n{}".format(payload)
    r = requests.post(creds.post_comment_url.format(community_name),
                      headers=headers, data=json.dumps(payload))


if __name__ == "__main__":
    auth_token = fetch_auth_token()
    secret_id = '56b9565ae4b01c56828d8584'
    post_comment(secret_id, auth_token, 'Testing HTTP Post')

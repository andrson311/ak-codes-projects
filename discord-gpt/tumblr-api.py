import os
import json
import pytumblr2
from requests_oauthlib import OAuth1Session

ROOT = os.path.dirname(__file__)

def get_tokens():
    request_token_url = 'http://www.tumblr.com/oauth/request_token'
    authorize_url = 'http://www.tumblr.com/oauth/authorize'
    access_token_url = 'http://www.tumblr.com/oauth/access_token'

    with open(os.path.join(ROOT, 'auth.json'), 'r') as auth_file:
        auth_data = json.load(auth_file)
        consumer_key = auth_data['tumblr-tokens']['consumer-key']
        consumer_secret = auth_data['tumblr-tokens']['consumer-secret']

    oauth_session = OAuth1Session(consumer_key, client_secret=consumer_secret)
    fetch_response = oauth_session.fetch_request_token(request_token_url)
    resource_owner_key = fetch_response.get('oauth_token')
    resource_owner_secret = fetch_response.get('oauth_token_secret')

    full_authorize_url = oauth_session.authorization_url(authorize_url)

    if 'redirect-response' not in auth_data['tumblr-tokens']:
        print(f'\nPlease go here and authorize: \n{full_authorize_url}')
        redirect_response = input('Allow then paste the full redirect URL here:\n').strip()
        with open(os.path.join(ROOT, 'auth.json'), 'w') as auth_file:
            auth_data['tumblr-tokens']['redirect-response'] = redirect_response
            json.dump(auth_data, auth_file, indent=4)
    else:
        redirect_response = auth_data['tumblr-tokens']['redirect-response']

    if 'oauth-token' not in auth_data['tumblr-tokens']:
        oauth_response = oauth_session.parse_authorization_response(redirect_response)
        verifier = oauth_response.get('oauth_verifier')

        oauth_session = OAuth1Session(
            consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=resource_owner_key,
            resource_owner_secret=resource_owner_secret,
            verifier=verifier
        )

        print(verifier)
        oauth_tokens = oauth_session.fetch_access_token(access_token_url)

        with open(os.path.join(ROOT, 'auth.json'), 'w') as auth_file:
            auth_data['tumblr-tokens']['oauth-token'] = oauth_tokens.get('oauth_token')
            auth_data['tumblr-tokens']['oauth-token-secret'] = oauth_tokens.get('oauth_token_secret')
            json.dump(auth_data, auth_file, indent=4)
            oauth_token = auth_data['tumblr-tokens']['oauth-token']
            oauth_token_secret = auth_data['tumblr-tokens']['oauth-token-secret']
    else:
        oauth_token = auth_data['tumblr-tokens']['oauth-token']
        oauth_token_secret = auth_data['tumblr-tokens']['oauth-token-secret']


    return {
        'consumer_key': consumer_key,
        'consumer_secret': consumer_secret,
        'oauth_token': oauth_token,
        'oauth_token_secret': oauth_token_secret
    }

tokens = get_tokens()
client = pytumblr2.TumblrRestClient(
    tokens['consumer_key'],
    tokens['consumer_secret'],
    tokens['oauth_token'],
    tokens['oauth_token_secret']
)

print(client.info())
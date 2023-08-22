import os
import json
import base64
import requests
import datetime
from datetime import timezone

ROOT = os.path.dirname(__file__)

def get_tokens(token_name):
    with open(os.path.join(ROOT, 'auth.json'), 'r') as auth_file:
        auth_data = json.load(auth_file)
        tokens = auth_data[token_name]
        return tokens
    
wp_access = get_tokens('wordpress')

wp_connection = wp_access['user'] + ':' + wp_access['key']
token = base64.b64encode(wp_connection.encode())

headers = {
    'Authorization': 'Basic ' + token.decode('utf-8')
}

api_url = 'http://127.0.0.1/wp-json/wp/v2/posts'

def get_posts(api_url):

    posts = []
    page = 1

    while True:
        response = requests.get(api_url, params={'page': page, 'per_page': 100})

        if not response.status_code == 200:
            break

        posts += response.json()
        page += 1

    return posts

response = get_posts(api_url)
for post in response:
    print(post['title']['rendered'])

print('Total posts:', len(response))

def create_new_post(new_post):

    response = requests.post(api_url, json=new_post, headers=headers)

    if response.status_code == 201:
        print(f'Just posted: {response.json()["title"]["rendered"]}')
    else:
        print('Oops, something went wrong.')

new_post = {
    'date': datetime.datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
    'title': 'This post was posted using python',
    'slug': 'this-is-python-post',
    'content': "<!-- wp:paragraph -->Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.<!-- /wp:paragraph -->",
    'status': 'publish'
}

create_new_post(new_post)


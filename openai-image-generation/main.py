import os
import json
import openai
import urllib.request
from urllib.parse import urlparse

ROOT = os.path.dirname(__file__)
def get_token(token_name):
    
    auth_file = open(os.path.join(ROOT, 'auth.json'))
    auth_data = json.load(auth_file)
    token = auth_data[token_name]
    return token

openai.api_key = get_token('openai-token')


prompt = 'Futuristic robot artist'

response = openai.Image.create(
    prompt=prompt,
    n=1,
    size='512x512'
)

image_url = response['data'][0]['url']

parsed_url = urlparse(image_url)
image_path = os.path.join(ROOT, os.path.basename(parsed_url.path))
urllib.request.urlretrieve(image_url, image_path)


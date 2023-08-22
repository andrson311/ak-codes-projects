import os
import json
from elevenlabs import generate, play, set_api_key, voices, save

ROOT = os.path.dirname(__file__)

def get_token(token_name):
    with open(os.path.join(ROOT, 'auth.json'), 'r') as auth_file:
        auth_data = json.load(auth_file)
        token = auth_data[token_name]
        return token

set_api_key(get_token('elevenlabs'))

audio = generate(
    text="Hello! My name is Glyph, nice to meet you!",
    voice="Emily",
    model='eleven_multilingual_v2'
)

play(audio)

output_file = os.path.join(ROOT, 'output.mp3')

save(audio, output_file)

# Different voice options
available_voices = voices()
for voice in available_voices:
    v = dict(voice)
    print(f'Name: {v["name"]}, Info: {v["labels"]} \n')

print(len(available_voices))


import os
import json
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist

from kaggle.api.kaggle_api_extended import KaggleApi

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from pprint import pprint

ROOT = os.path.dirname(__file__)

def get_tokens(api):
    with open(os.path.join(ROOT, 'auth.json'), 'r') as auth_file:
        auth_data = json.load(auth_file)
        tokens = auth_data[api]
        return tokens
    
spotify_tokens = get_tokens('spotify')

spotify_api = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=spotify_tokens['client_id'],
    client_secret=spotify_tokens['client_secret']
))

kaggle_api = KaggleApi()
kaggle_api.authenticate()

kaggle_api.dataset_download_files(
    'vatsalmavani/spotify-dataset',
    path=ROOT,
    unzip=True
)

data = pd.read_csv(os.path.join(ROOT, 'data', 'data.csv'))

X = data.select_dtypes(np.number)
number_cols = list(X.columns)

song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, verbose=False, n_init='auto'))
], verbose=False)

song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

def find_song(name, year):
    song_data = defaultdict()
    results = spotify_api.search(q=f'track: {name} year: {year}', limit=1)
    if results['tracks']['items'] == []:
        return None
    
    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = spotify_api.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value
    
    return pd.DataFrame(song_data)

def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, spotify_data):
    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    
    song_matrix = np.array(list(song_vectors), dtype=object)
    return np.mean(song_matrix, axis=0)

def flatten_dict_list(dict_list):

    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
        
    return flattened_dict

def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

songs = recommend_songs([{'name': 'ein auto, ein mann', 'year':2007}],  data)

pprint(songs)
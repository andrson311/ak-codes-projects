import os
import numpy as np
from scipy.io import wavfile
import librosa
import keras
from keras import layers
from keras import backend as K
from sklearn.model_selection import train_test_split
from glob import glob
from kaggle.api.kaggle_api_extended import KaggleApi

SEED = 42
ROOT = os.path.dirname(__file__)
AUDIO_PATH = os.path.join(ROOT, 'Data', 'genres_original')
BATCH_SIZE = 2
SAMPLING_RATE = 16000
DURATION = 20
AUDIO_SIZE = SAMPLING_RATE * DURATION
LATENT_DIM = 16
EPOCHS = 10

np.random.seed(seed=SEED)
api = KaggleApi()
api.authenticate()

if not os.path.exists(AUDIO_PATH):
    api.dataset_download_files(
        'andradaolteanu/gtzan-dataset-music-genre-classification',
        path=ROOT,
        unzip=True
    )

def load_data(cls):

    audio_files = glob(os.path.join(AUDIO_PATH, cls, '*.wav'))
    audio_data = []

    for path in audio_files:
        audio, sr = librosa.load(path, sr=SAMPLING_RATE, offset=0.0, duration=DURATION)
        audio_data.append(audio)

    audio_data = np.array(audio_data)
    audio_data = audio_data.reshape(audio_data.shape[0], AUDIO_SIZE, 1)
    input_shape = (AUDIO_SIZE, 1)
    audio_data = audio_data.astype('float32')

    return audio_data

data = load_data('jazz')
print(data)

inp = layers.Input(shape=(AUDIO_SIZE, 1), name='encoder_input')
cx = layers.Conv1D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(inp)
cx = layers.BatchNormalization()(cx)
cx = layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
cx = layers.BatchNormalization()(cx)
cx = layers.Conv1D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
cx = layers.BatchNormalization()(cx)
x = layers.Flatten()(cx)
x = layers.Dense(2 * LATENT_DIM, activation='relu')(x)
x = layers.BatchNormalization()(x)
mu = layers.Dense(LATENT_DIM, name='latent_mu')(x)
sigma = layers.Dense(LATENT_DIM, name='latent_sigma')(x)

conv_shape = K.int_shape(cx)
print('Shape of last convolutional layer:', conv_shape)

def sample_z(args):
    mu, sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch, dim))
    return mu + K.exp(sigma / 2) * eps


z = layers.Lambda(sample_z, output_shape=(LATENT_DIM, ), name='z')([mu, sigma])

encoder = keras.Model(inp, [mu, sigma, z], name='encoder')
print(encoder.summary())

d_inp = layers.Input(shape=(LATENT_DIM, ), name='decoder_input')
x = layers.Dense(conv_shape[1] * conv_shape[2], activation='relu')(d_inp)
x = layers.BatchNormalization()(x)
x = layers.Reshape((conv_shape[1], conv_shape[2]))(x)
cx = layers.Conv1DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
cx = layers.BatchNormalization()(cx)
cx = layers.Conv1DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
cx = layers.BatchNormalization()(cx)
cx = layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
cx = layers.BatchNormalization()(cx)
output = layers.Conv1DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)

decoder = keras.Model(d_inp, output, name='decoder')
print(decoder.summary())

vae_outputs = decoder(encoder(inp)[2])
vae = keras.Model(inp, vae_outputs, name='vae')
print(vae.summary())

def kl_reconstruction_loss(true, pred):
    reconstruction_loss = keras.losses.binary_crossentropy(K.flatten(true), K.flatten(pred)) * AUDIO_SIZE
    kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    return K.mean(reconstruction_loss + kl_loss)

vae.compile(optimizer='adam', loss=kl_reconstruction_loss)
vae.fit(data, data, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
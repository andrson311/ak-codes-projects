import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import librosa
import soundfile as sf
from glob import glob
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

SEED = 42
ROOT = os.path.dirname(__file__)
AUDIO_PATH = os.path.join(ROOT, 'Data', 'genres_original')
BATCH_SIZE = 1
SAMPLING_RATE = 4000
DURATION = 5
AUDIO_SIZE = SAMPLING_RATE * DURATION
LATENT_DIM = 50
EPOCHS = 20

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
    audio_data = audio_data.astype('float32')

    return audio_data

data = load_data('jazz')
print(data)

def sample_z(args):
    mu, sigma = args
    batch = tf.shape(mu)[0]
    dim = mu.shape[1]
    eps = tf.random.normal(shape=(batch, dim))
    return mu + tf.math.exp(sigma / 2) * eps

inp = tf.keras.layers.Input(shape=(AUDIO_SIZE, 1), name='encoder_input')
cx = tf.keras.layers.Conv1D(filters=128, kernel_size=256, strides=50, padding='same', activation='relu')(inp)
#cx = tfa.layers.InstanceNormalization()(cx)
cx = tf.keras.layers.Conv1D(filters=256, kernel_size=256, strides=50, padding='same', activation='relu')(cx)
#cx = tfa.layers.InstanceNormalization()(cx)
#cx = tf.keras.layers.Conv1D(filters=1, kernel_size=SAMPLING_RATE // 2, strides=2, padding='same', activation='relu')(cx)
#cx = tfa.layers.InstanceNormalization()(cx)
x = tf.keras.layers.Flatten()(cx)
#x = tf.keras.layers.Dense(2 * LATENT_DIM, activation='relu')(x)
mu = tf.keras.layers.Dense(LATENT_DIM, name='latent_mu')(x)
sigma = tf.keras.layers.Dense(LATENT_DIM, name='latent_sigma')(x)
z = tf.keras.layers.Lambda(sample_z, output_shape=(LATENT_DIM, ), name='z')([mu, sigma])

encoder = tf.keras.Model(inp, [mu, sigma, z], name='encoder')
print(encoder.summary())

conv_shape = cx.shape

d_inp = tf.keras.layers.Input(shape=(LATENT_DIM, ), name='decoder_input')
x = tf.keras.layers.Dense(conv_shape[1] * conv_shape[2], activation='relu')(d_inp)
x = tf.keras.layers.Reshape((conv_shape[1], conv_shape[2]))(x)
cx = tf.keras.layers.Conv1DTranspose(filters=256, kernel_size=256, strides=50, padding='same', activation='relu')(x)
#cx = tf.keras.layers.BatchNormalization()(cx)
cx = tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=256, strides=50, padding='same', activation='relu')(cx)
#cx = tf.keras.layers.BatchNormalization()(cx)
output = tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=256, strides=1, 
                                         padding='same', name='decoder_output')(cx)

decoder = tf.keras.Model(d_inp, output, name='decoder')
print(decoder.summary())

vae_outputs = decoder(encoder(inp)[2])
vae = tf.keras.Model(inp, vae_outputs, name='vae')
print(vae.summary())

def kl_reconstruction_loss(true, pred):
    reconstruction_loss = tf.keras.losses.binary_crossentropy(tf.squeeze(true), tf.squeeze(pred)) * AUDIO_SIZE
    kl_loss = 1 + sigma - tf.math.square(mu) - tf.math.exp(sigma)
    kl_loss = tf.math.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    return tf.math.reduce_mean(reconstruction_loss + kl_loss)

vae.compile(optimizer='adam', loss=kl_reconstruction_loss)
vae.fit(data, data, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
vae.save(os.path.join(ROOT, 'saved_model/VAE'))
encoder.save(os.path.join(ROOT, 'saved_model/Encoder'))
decoder.save(os.path.join(ROOT, 'saved_model/Decoder'))

decoder = tf.keras.models.load_model(os.path.join(ROOT, 'saved_model/Decoder'))
encoder = tf.keras.models.load_model(os.path.join(ROOT, 'saved_model/Encoder'))
#gen = np.random.normal(0, 1, size=(1, LATENT_DIM))

gen = np.random.rand(1, LATENT_DIM)
reconstruct = decoder.predict(encoder.predict(data[:1], steps=1)[2])
pred = decoder.predict(gen, steps=1)
example = np.array(np.squeeze(data[0]))
reconstruct = np.array(np.squeeze(reconstruct[0]))
pred = np.array(np.squeeze(pred))

reconstruct /= np.linalg.norm(reconstruct)
pred /= np.linalg.norm(pred)

print(example)
print(pred.shape, pred)
print(reconstruct.shape, reconstruct)


sf.write(os.path.join(ROOT, 'example.wav'), example, SAMPLING_RATE, subtype='PCM_16')
sf.write(os.path.join(ROOT, 'generated.wav'), pred, SAMPLING_RATE, subtype='PCM_16')
sf.write(os.path.join(ROOT, 'reconstructed.wav'), reconstruct, SAMPLING_RATE, subtype='PCM_16')
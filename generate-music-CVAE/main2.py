import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import scipy
from keras import layers
from sklearn.model_selection import train_test_split
from glob import glob
from kaggle.api.kaggle_api_extended import KaggleApi

SEED = 42
ROOT = os.path.dirname(__file__)
AUDIO_PATH = os.path.join(ROOT, 'Data', 'genres_original')
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 2
SAMPLING_RATE = 16000
DURATION = 20
AUDIO_SIZE = SAMPLING_RATE * DURATION
LATENT_DIM = 16
EPOCHS = 10

tf.random.set_seed(seed=SEED)
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
        raw_audio = tf.io.read_file(path)
        try:
            waveform, _ = tf.audio.decode_wav(raw_audio)
            waveform = waveform[:AUDIO_SIZE]
            audio_data.append(waveform)
        except:
            continue

    train_set, val_set = train_test_split(audio_data, test_size=0.2, random_state=SEED)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_set
            ).cache().prefetch(AUTOTUNE).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_set
            ).cache().prefetch(AUTOTUNE).batch(BATCH_SIZE)

    return train_dataset, val_dataset

def build_encoder():

    model = tf.keras.Sequential([
        layers.Conv1D(filters=32, kernel_size=3, strides=2, padding='same', 
                      input_shape=(AUDIO_SIZE, 1), activation='relu'),
        layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.4),
        layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.4),
        layers.Conv1D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.Conv1D(filters=128, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.4),
        layers.Conv1D(filters=128, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.4),
        layers.Conv1D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.Conv1D(filters=256, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.4),
        layers.Conv1D(filters=256, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.4),
        layers.Flatten(),
        layers.Dense(2 * LATENT_DIM)
    ])

    return model

def build_decoder():

    model = tf.keras.Sequential([
        layers.Dense(units=625 * 256, activation='relu', input_shape=(LATENT_DIM,)),
        layers.Reshape(target_shape=(625, 256)),
        layers.Conv1DTranspose(filters=256, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.4),
        layers.Conv1DTranspose(filters=256, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.4),
        layers.Conv1DTranspose(filters=256, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.Conv1DTranspose(filters=128, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.4),
        layers.Conv1DTranspose(filters=128, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.4),
        layers.Conv1DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.Conv1DTranspose(filters=64, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.4),
        layers.Conv1DTranspose(filters=64, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.4),
        layers.Conv1DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.Conv1DTranspose(filters=1, kernel_size=3, strides=1, padding='same')
    ])

    return model

class CVAE(tf.keras.Model):
    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder = build_encoder()
        self.decoder = build_decoder()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(1, LATENT_DIM))
        return self.decode(eps, apply_sigmoid=True)
    
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis
        )
    
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            x_logit = self.decode(z)
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
            logpz = self.log_normal_pdf(z, 0., 0.)
            logqz_x = self.log_normal_pdf(z, mean, logvar)
            loss_KL = -tf.reduce_mean(logpx_z + logpz - logqz_x)
            reconstruction_loss = tf.reduce_mean(
                     tf.keras.losses.binary_crossentropy(x, x_logit)
                 )
            total_loss = reconstruction_loss + loss_KL

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return total_loss
        

    def train(self, train_dataset, val_dataset):
        for epoch in range(EPOCHS):
            batch_count = 0
            mean = tf.keras.metrics.Mean()
            for train_batch in train_dataset:
                loss = self.train_step(train_batch)
                print(f'epoch: {epoch}, batch: {batch_count}/{len(train_dataset)}, loss: {loss}', end='\r')
                batch_count += 1
            
            print()
            mean.reset_state()
            for val_batch in val_dataset:
                batch_loss = self.compute_loss(val_batch)
                mean(batch_loss)
                
            elbo = -mean.result()
            print(f'epoch: {epoch}, ELBO: {elbo}')



train_dataset, val_dataset = load_data('jazz')
model = CVAE()
model.train(train_dataset, val_dataset)

generated = model.sample()
generated = np.squeeze(generated) * np.iinfo(np.int16).max
print(generated)
scipy.io.wavfile.write('generated.wav', SAMPLING_RATE, generated.astype(np.int16))

pred_count = 0
for test in val_dataset:
    mean, logvar = model.encode(test)
    z = model.reparameterize(mean, logvar)
    prediction = model.sample(z)
    prediction = np.squeeze(prediction) * np.iinfo(np.int16).max
    scipy.io.wavfile.write(os.path.join(ROOT, f'predicted_{pred_count}.wav'), SAMPLING_RATE, generated.astype(np.int16))
    pred_count += 1


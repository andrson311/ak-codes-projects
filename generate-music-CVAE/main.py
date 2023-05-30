import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import librosa
from glob import glob
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from keras import layers

api = KaggleApi()
api.authenticate()

SEED = 42
ROOT = os.path.dirname(__file__)
AUDIO_PATH = os.path.join(ROOT, 'Data', 'genres_original')
AUTOTUNE = tf.data.AUTOTUNE
SAMPLING_RATE = 16000
DURATION = 30
AUDIO_SIZE = SAMPLING_RATE * DURATION + 1
BATCH_SIZE = 4
LATENT_DIM = 32
EPOCHS = 20

if not os.path.exists(AUDIO_PATH):
    api.dataset_download_files(
        'andradaolteanu/gtzan-dataset-music-genre-classification',
        path=ROOT,
        unzip=True
    )

def load_file(filename):
    data, sr = librosa.load(filename, sr=SAMPLING_RATE, offset=0.0, duration=DURATION)
    print(data)
    data = data.reshape(1, AUDIO_SIZE)
    return data

def load_data(cls):
    """audio_files =  glob(os.path.join(AUDIO_PATH, cls, '*.wav'))

    train_set, test_set = train_test_split(audio_files, test_size=0.2, random_state=SEED)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_set
    ).map(
        map_func=lambda x: tf.py_function(load_file, [x], [tf.float32])
    ).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_set)
    ).map(
        map_func=lambda x: tf.py_function(load_file, [x], [tf.float32])
    ).batch(BATCH_SIZE) """

    train_dataset, test_dataset = tf.keras.utils.audio_dataset_from_directory(
        directory=os.path.join(AUDIO_PATH, cls),
        labels=None,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        seed=SEED,
        output_sequence_length=AUDIO_SIZE,
        subset='both'
    )

    return train_dataset, test_dataset

class Resnet1DBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, type='encode'):
        super(Resnet1DBlock, self).__init__()

        if type == 'encode':
            self.conv1a = layers.Conv1D(filters, kernel_size, 2, padding='same')
            self.conv1b = layers.Conv1D(filters, kernel_size, 1, padding='same')
            self.norm1a = tfa.layers.InstanceNormalization()
            self.norm1b = tfa.layers.InstanceNormalization()
        
        elif type == 'decode':
            self.conv1a = layers.Conv1DTranspose(filters, kernel_size, 1, padding='same')
            self.conv1b = layers.Conv1DTranspose(filters, kernel_size, 1, padding='same')
            self.norm1a = layers.BatchNormalization()
            self.norm1b = layers.BatchNormalization()
        
        else:
            return None
        
    def call(self, input_tensor):

        x = tf.nn.relu(input_tensor)
        x = self.conv1a(x)
        x = self.norm1a(x)
        x = layers.LeakyReLU(0.4)(x)
        x = self.conv1b(x)
        x = self.norm1b(x)
        x = layers.LeakyReLU(0.4)(x)

        return tf.nn.relu(x)

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(AUDIO_SIZE, 1)),
            layers.Conv1D(64, 1, 2),
            Resnet1DBlock(64, 1),
            layers.Conv1D(128, 1, 2),
            Resnet1DBlock(128, 1),
            layers.Conv1D(256, 1, 2),
            Resnet1DBlock(256, 1),
            layers.Conv1D(512, 1, 2),
            Resnet1DBlock(512, 1),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim)
        ])

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Reshape(target_shape=(1, latent_dim)),
            Resnet1DBlock(512, 1, 'decode'),
            layers.Conv1DTranspose(512, 1, 1),
            Resnet1DBlock(256, 1, 'decode'),
            layers.Conv1DTranspose(256, 1, 1),
            Resnet1DBlock(128, 1, 'decode'),
            layers.Conv1DTranspose(128, 1, 1),
            Resnet1DBlock(64, 1, 'decode'),
            layers.Conv1DTranspose(64, 1, 1),
            layers.Conv1DTranspose(AUDIO_SIZE, 1)
        ])

    @tf.function
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(200, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

@tf.function
def log_normal_pdf(sample, mean, logvar, axis):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)
    )

@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    logits = model.decode(z)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x)
    logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x), logits


@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        
        loss_KL, logits = compute_loss(model, x)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(x, logits)
        )
        total_loss = loss_KL + reconstruction_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss

def train(train_dataset, test_dataset, model, epochs):
    for epoch in range(1, epochs + 1):
        for batch in train_dataset:
            batch = np.asarray(batch)[0]
            train_loss = train_step(model, batch, optimizer)
        
        val_loss = tf.keras.metrics.Mean()
        for batch in test_dataset:
            batch = np.asarray(batch)[0]
            loss_KL, _ = compute_loss(model, batch)
            val_loss(loss_KL)
        
        print(f'epoch: {epoch}, loss: {train_loss}, val_loss: {val_loss}', end='\r')



train_set, test_set = load_data('jazz')
optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model = CVAE(latent_dim=LATENT_DIM)
train(train_set, test_set, model, EPOCHS)


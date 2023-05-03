import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras import layers
import numpy as np
from glob import glob
import os
import PIL

# constants
root = os.path.dirname(__file__)
MONET_FILES = glob(os.path.join(root, 'data', 'monet_tfrec', '*.tfrec'))
PHOTO_FILES = glob(os.path.join(root, 'data', 'photo_tfrec', '*.tfrec'))
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = 256
OUTPUT_CHANNELS = 3
LAMBDA_CYCLE = 10

# decode individual image
def load_image(example):
    tfrecord_format = {
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    return image

# load dataset from TFRecord files
def load_dataset(files):
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    return dataset

# group of layers for downsampling
def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result

# group of layers for upsampling
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))
    
    result.add(layers.ReLU())

    return result

# function for creating a generator model
def create_generator_model():
    inputs = layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3])

    down_stack = [
        downsample(64, 4, apply_instancenorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)

    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

    x = inputs

    skips = []

    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)

# function for creating discriminator model
def create_discriminator_model():

    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inputs = layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name='input_image')

    x = inputs
    x = downsample(64, 4, False)(x)
    x = downsample(128, 4)(x)
    x = downsample(256, 4)(x)
    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = layers.LeakyReLU()(x)
    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)

    return keras.Model(inputs=inputs, outputs=x)


# define necessary loss functions
def generator_loss(generated):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    loss = cross_entropy(tf.ones_like(generated), generated)
    return loss

def discriminator_loss(real, generated):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        real_loss = cross_entropy(tf.ones_like(real), real)
        generated_loss = cross_entropy(tf.zeros_like(generated), generated)

        loss = real_loss + generated_loss

        return loss * 0.5

def cycle_loss(real_image, cycled_image):
    loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    
    return loss * LAMBDA_CYCLE

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))

    return loss * LAMBDA_CYCLE * 0.5

# define cycle GAN model class
class CycleGAN(keras.Model):
    def __init__(self, monet_generator, photo_generator, monet_discriminator, photo_discriminator):
        super(CycleGAN, self).__init__()

        self.monet_generator = monet_generator
        self.photo_generator = photo_generator
        self.monet_discriminator = monet_discriminator
        self.photo_discriminator = photo_discriminator
    
    def compile(self, monet_gen_optimizer, photo_gen_optimizer, monet_disc_optimizer, photo_disc_optimizer,
                gen_loss_fn, disc_loss_fn, cycle_loss_fn, identity_loss_fn):
        super(CycleGAN, self).compile()
        self.monet_gen_optimizer = monet_gen_optimizer
        self.photo_gen_optimizer = photo_gen_optimizer
        self.monet_disc_optimizer = monet_disc_optimizer
        self.photo_disc_optimizer = photo_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
    
    def train_step(self, batch_data):
        real_monet, real_photo = batch_data

        with tf.GradientTape(persistent=True) as tape:
            fake_monet = self.monet_generator(real_photo, training=True)
            cycled_photo = self.photo_generator(fake_monet, training=True)

            fake_photo = self.photo_generator(real_monet, training=True)
            cycled_monet = self.monet_generator(fake_photo, training=True)

            same_monet = self.monet_generator(real_monet, training=True)
            same_photo = self.photo_generator(real_photo, training=True)

            disc_real_monet = self.monet_discriminator(real_monet, training=True)
            disc_real_photo = self.photo_discriminator(real_photo, training=True)

            disc_fake_monet = self.monet_discriminator(fake_monet, training=True)
            disc_fake_photo = self.photo_discriminator(fake_photo, training=True)

            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            monet_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet)
            photo_cycle_loss = self.cycle_loss_fn(real_photo, cycled_photo)
            total_cycle_loss = monet_cycle_loss + photo_cycle_loss

            monet_identity_loss = self.identity_loss_fn(real_monet, same_monet)
            photo_identity_loss = self.identity_loss_fn(real_photo, same_photo)

            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + monet_identity_loss
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + photo_identity_loss

            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

            monet_gen_gradients = tape.gradient(total_monet_gen_loss, self.monet_generator.trainable_variables)
            photo_gen_gradients = tape.gradient(total_photo_gen_loss, self.photo_generator.trainable_variables)

            monet_disc_gradients = tape.gradient(monet_disc_loss, self.monet_discriminator.trainable_variables)
            photo_disc_gradients = tape.gradient(photo_disc_loss, self.photo_discriminator.trainable_variables)

            self.monet_gen_optimizer.apply_gradients(zip(monet_gen_gradients, self.monet_generator.trainable_variables))
            self.photo_gen_optimizer.apply_gradients(zip(photo_gen_gradients, self.photo_generator.trainable_variables))
            self.monet_disc_optimizer.apply_gradients(zip(monet_disc_gradients, self.monet_discriminator.trainable_variables))
            self.photo_disc_optimizer.apply_gradients(zip(photo_disc_gradients, self.photo_discriminator.trainable_variables))

            return {
                "monet_gen_loss": total_monet_gen_loss,
                "photo_gen_loss": total_photo_gen_loss,
                "monet_disc_loss": monet_disc_loss,
                "photo_disc_loss": photo_disc_loss
            }

# load and preprocess data
monet_ds = load_dataset(MONET_FILES).batch(1)
photo_ds = load_dataset(PHOTO_FILES).batch(1)

# define models
monet_generator = create_generator_model()
photo_generator = create_generator_model()
monet_discriminator = create_discriminator_model()
photo_discriminator = create_discriminator_model()

# define optimizers
monet_gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
photo_gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
monet_disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
photo_disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# create cycle GAN model, compile, and train it
model = CycleGAN(
    monet_generator=monet_generator,
    photo_generator=photo_generator,
    monet_discriminator=monet_discriminator,
    photo_discriminator=photo_discriminator
)   

model.compile(
    monet_gen_optimizer=monet_gen_optimizer,
    photo_gen_optimizer=photo_gen_optimizer,
    monet_disc_optimizer=monet_disc_optimizer,
    photo_disc_optimizer=photo_disc_optimizer,
    gen_loss_fn=generator_loss,
    disc_loss_fn=discriminator_loss,
    cycle_loss_fn=cycle_loss,
    identity_loss_fn=identity_loss,
)

model.fit(
    tf.data.Dataset.zip((monet_ds, photo_ds)),
    epochs=25
)

# transform all photographs and save them locally
i = 1
for img in photo_ds:
    prediction = monet_generator(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    im = PIL.Image.fromarray(prediction)
    im.save(os.path.join('transformed', str(i) + '.jpg'))
    i += 1




    


import tensorflow as tf
import numpy as np
import os
from PIL import Image

root = os.path.dirname(__file__)
image_path = os.path.join(root, 'photo.jpg')

def load_image(image_path, max_dim=None):
    image = Image.open(image_path)

    if max_dim:
        image.thumbnail((max_dim, max_dim))
    
    return np.array(image)

def deprocess(image):
    image = 255 * (image + 1.0) / 2.0
    return tf.cast(image, tf.uint8)

def calc_loss(image, model):
    image_batch = tf.expand_dims(image, axis=0)
    layer_activations = model(image_batch)

    if len(layer_activations) == 1:
        layer_activations = [layer_activations]
    
    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)
    
    return tf.reduce_sum(losses)

class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model
    
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32)
        )
    )
    def __call__(self, image, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for _ in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(image)
                loss = calc_loss(image, self.model)
            
            gradients = tape.gradient(loss, image)
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            image += gradients * step_size
            image = tf.clip_by_value(image, -1, 1)

        return loss, image

base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
print(base_model.summary())
names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
deepdream = DeepDream(dream_model)

def run_deep_dream(image, steps=100, step_size=0.01):
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    image = tf.convert_to_tensor(image)
    step_size = tf.convert_to_tensor(step_size)
    
    steps_remaining = steps
    step = 0
    while steps_remaining:
        run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps
        loss, image = deepdream(image, run_steps, tf.constant(step_size))

        print("Step {}, loss {}".format(step, loss))
    
    result = deprocess(image)

    return result

original_image = load_image(image_path=image_path, max_dim=512)
dream_image = run_deep_dream(image=original_image, steps=100, step_size=0.01)
dream_image = Image.fromarray(np.array(dream_image))
dream_image.save(os.path.join(root, 'dream.jpg'))






import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   
import keras_cv
from PIL import Image

root = os.path.dirname(__file__)
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

n_images = 5

for i in range(n_images):

    images = model.text_to_image("beautiful mockup for a plain t-shirt", batch_size=1)

    Image.fromarray(images[0]).save(os.path.join(root, f'generated_{i}.jpg'))

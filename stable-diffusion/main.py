import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   
import keras_cv
from tensorflow import keras
from PIL import Image
import os

root = os.path.dirname(__file__)
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

images = model.text_to_image("magical land", batch_size=1)

count = 0
for img in images:
    Image.fromarray(img).save(os.path.join(root, f'generated_{count}.jpg'))
    count += 1

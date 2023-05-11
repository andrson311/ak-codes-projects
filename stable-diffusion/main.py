import keras_cv
from tensorflow import keras
from PIL import Image

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

images = model.text_to_image("photograph of an astronaut riding a horse", batch_size=1)

for img in images:
    image = Image.open(img)
    image.save('generated.jpg')
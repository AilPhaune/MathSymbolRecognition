import sys
import numpy as np
import glob
import os.path as path
from termcolor import cprint

_, image_path, model_path = sys.argv

if image_path.endswith("/"):
    images_path = glob.glob(image_path + "**/*.png")
else:
    images_path = [image_path]

print("Images path:", images_path)
print("Model path:", model_path)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

img_height, img_width = 64, 64

img_array = []
for image_path in images_path:
    img = load_img(
        image_path, target_size=(img_height, img_width)
    )
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img_array.append(img)

img_array = np.vstack(img_array)

model = load_model(model_path)
model.summary()

predictions = model.predict(img_array)

for prediction, img_path in zip(predictions, images_path):
    score = tf.nn.softmax(prediction)

    category = np.argmax(score)
    dir_name = path.basename(path.dirname(img_path))

    cprint(f"Model thinks {img_path} is {category} (in reality: {dir_name}) with {100*np.max(score)}% certainty", "green" if dir_name == str(category) else "red")
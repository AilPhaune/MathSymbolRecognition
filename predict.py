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

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import json

classes_path = model_path.replace("classifier","classes").replace(".keras",".json")

with open(classes_path, "r") as f:
    classes_names = json.loads(f.read())

img_height, img_width = 64, 64

img_array = []
for image_path in images_path:
    img = load_img(
        image_path, target_size=(img_height, img_width), color_mode='grayscale'
    )
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img_array.append(img)

img_array = np.vstack(img_array)

model = load_model(model_path)
model.summary()

predictions = model.predict(img_array)

ok, nok = 0, 0

print()
print("-"*100)
print()

validation = "validation_data" in image_path

for prediction, img_path in zip(predictions, images_path):
    score = tf.nn.softmax(prediction)

    category = classes_names[np.argmax(score)]
    dir_name = path.basename(path.dirname(img_path))

    sorted_categories = sorted(list(range(len(classes_names))), key=lambda x: score[x], reverse=True)
    
    if dir_name != category and validation:
        cprint(f"Model thinks {img_path} is {category} (in reality: {dir_name}) with {100*np.max(score)}% certainty", "red")
        print("Order of preference:", [classes_names[k] for k in sorted_categories])
        print(score)
        nok += 1
    elif not validation:
        cprint(f"Model thinks {img_path} is {category} with {100*np.max(score)}% certainty", "yellow")
        print("Order of preference:", [classes_names[k] for k in sorted_categories])
        print(score)
    else:
        ok += 1

if validation:
    cprint(f"Model correctly classifies {ok}/{ok+nok} images ({100.*ok/(ok+nok)}%)", "green")
    cprint(f"Model incorrectly classifies {nok}/{ok+nok} images ({100.*nok/(ok+nok)}%)", "red")
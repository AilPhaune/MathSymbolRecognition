import sys
import numpy as np
import glob
import os.path as path
import datetime
from termcolor import cprint

_, image_path, model_path, *_ = sys.argv

if image_path.endswith("/"):
    images_path = glob.glob(image_path + "**/*.png")
else:
    images_path = [image_path]

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.config import enable_unsafe_deserialization
import json

classes_path = model_path.replace("classifier","classes").replace(".keras",".json")

### Dangerous:
if "--unsafe-deserialize" in sys.argv:
    if "--source-trusted" not in sys.argv:
        assert input("Do not load models with unsafe deserialization from sources you don't trust.\nType 'Y' in caps and press ENTER to continue or anything else to exit.") == "Y"
    enable_unsafe_deserialization()
##############

with open(classes_path, "r") as f:
    classes_names = list(json.loads(f.read()))
    num_classes = len(classes_names)

img_height, img_width = 64, 64

img_array = []
for img_path in images_path:
    img = load_img(
        img_path, target_size=(img_height, img_width), color_mode='grayscale'
    )
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img_array.append(img)

img_array = np.vstack(img_array)

model = load_model(model_path)
model.summary()

predictions = model.predict(img_array, batch_size=32)

ok, nok, kok = 0, 0, 0

print()
print("-"*100)
print()

validation = "validation_data" in image_path

if validation:
    for category in classes_names:
        catpath = f"{image_path}{category}"
        if not path.isdir(catpath):
            cprint(f"No test folder for category {category} -- {catpath}", "yellow")

bad_categories = []

prediction_stats = {classes_names[i]:{classes_names[j]:0 for j in range(num_classes)} for i in range(num_classes)}

for score, img_path in zip(predictions, images_path):
    # the real category
    dir_name = path.basename(path.dirname(img_path))

    if dir_name not in classes_names and validation:
        if dir_name not in bad_categories:
            bad_categories.append(dir_name)
        continue

    # the predicted category
    category_id = np.argmax(score)
    category = classes_names[category_id]

    if validation:
        prediction_stats[dir_name][category] += 1

    sorted_categories = sorted(list(range(num_classes)), key=lambda x: score[x], reverse=True)
    
    if dir_name != category and validation:
        cprint(f"Model thinks {img_path} is {category} (in reality: {dir_name}) with {100*np.max(score)}% certainty", "red")
        print("Order of preference:")
        for i in sorted_categories[0:3]:
            print(f'{classes_names[i]} ({i}): {int(score[i]*1000)/10}%')
        print()
        nok += 1
    elif sorted_categories[1] == dir_name and validation:
        cprint(f"Model thinks {img_path} is {category} (in reality: {dir_name}) with {100*np.max(score)}% certainty", "yellow")
        print("Order of preference:")
        for i in sorted_categories[0:3]:
            print(f'{classes_names[i]}: {int(score[i]*1000)/10}%')
        print()
        kok += 1
    elif not validation:
        cprint(f"Model thinks {img_path} is {category} with {100*np.max(score)}% certainty", "yellow")
        print("Order of preference:")
        for i in sorted_categories[0:3]:
            print(f'{classes_names[i]}: {int(score[i]*1000)/10}%')
    else:
        ok += 1

for bad in bad_categories:
    cprint(f"Category {bad} is not in this model's capacity !", "cyan")

if validation:
    cprint(f"Model correctly classifies {ok}/{ok+nok+kok} images ({100.*ok/(ok+nok+kok)}%)", "green")
    cprint(f"Model classifies as the second choice {kok}/{ok+nok+kok} images ({100.*kok/(ok+nok+kok)}%)", "yellow")
    cprint(f"Model incorrectly classifies {nok}/{ok+nok+kok} images ({100.*nok/(ok+nok+kok)}%)", "red")

    now = datetime.datetime.now()
    with open(f'prediction_stats/predict_{now.strftime("%d-%m-%Y_%H-%M")}.html', "w") as f:
        with open("stats_viewer_template.html", "r") as template:
            f.write(template.read().replace('//', f'makeTable({json.dumps(prediction_stats)});'))
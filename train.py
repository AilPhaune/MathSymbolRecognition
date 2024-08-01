import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, Input, RandomRotation, RandomZoom, Dropout, RandomCrop, RandomTranslation, Lambda
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from termcolor import cprint
import sys
import random
import json
from glob import glob

epochs = 50
if len(sys.argv) > 1:
    epochs = int(sys.argv[1])

# Define constants
data_dir = 'dataset'
img_height = 64
img_width = 64
batch_size = 32

seed = random.randint(1, 2147483647)

# Generate training and validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='categorical',
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='categorical',
)

# Print class names
class_names = train_ds.class_names
cprint(f"Class Names: {class_names}", color="yellow")

# Data augmentation
data_augmentation = Sequential(
    [
        Input((img_height, img_width, 1)),
        RandomTranslation(0.04, 0.04),
        RandomRotation(0.025),
        RandomZoom(0.1),
        RandomTranslation(0.04, 0.04),
        RandomZoom(0.1),
        RandomTranslation(0.04, 0.04),
    ]
)

# Define the model
model = Sequential([
    Input((img_height, img_width, 1)),
    data_augmentation,

    Conv2D(32, 5, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.05),

    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='leaky_relu'),
    Dropout(0.1),

    Dense(len(class_names), activation='softmax')
])

# Compile the model
# model.compile(optimizer='rmsprop', loss=CategoricalCrossentropy(), metrics=['accuracy'])
model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[
        EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=False),
        ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5, verbose=1, min_lr=0.000001)
    ],
)

# Save the model
now = datetime.datetime.now()
model_path = f'models/classifier_{now.strftime("%d-%m-%Y_%H-%M")}.keras'
classes_path = f'models/classes_{now.strftime("%d-%m-%Y_%H-%M")}.json'

with open(classes_path, "w") as f:
    f.write(json.dumps(class_names))

model.save(model_path)
print(f'Model saved as {model_path}')
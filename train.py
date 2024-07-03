import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, Input, RandomRotation, RandomZoom, Dropout, RandomCrop, RandomTranslation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
batch_size = 64

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
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
)

# Print class names
class_names = train_ds.class_names
cprint(f"Class Names: {class_names}", color="yellow")

# Data augmentation
data_augmentation = Sequential(
    [
        Input((img_height, img_width, 1)),
        RandomTranslation(0.08, 0.08),
        RandomRotation(0.05),
        RandomZoom(0.1),
        RandomTranslation(0.08, 0.08),
        RandomRotation(0.05),
        RandomZoom(0.1),
        RandomTranslation(0.08, 0.08),
    ]
)

# Define the model
model = Sequential([
    Input((img_height, img_width, 1)),
    data_augmentation,

    Rescaling(1./255),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),

    Flatten(),

    Dropout(0.2),
    Dense(512, activation='sigmoid'),
    Dense(256, activation='sigmoid'),
    Dropout(0.1),

    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[
        EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=False),
        ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=10, verbose=1, min_lr=0.00001)
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
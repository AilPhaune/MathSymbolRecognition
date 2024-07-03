import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, Input, RandomRotation, RandomZoom, Dropout, RandomCrop, RandomTranslation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
data_dir = 'dataset'
img_height = 64
img_width = 64
batch_size = 16
epochs = 500

# Generate training and validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Print class names
class_names = train_ds.class_names
print("Class Names:", class_names)

# Data augmentation
data_augmentation = Sequential(
    [
        Input((img_height, img_width, 3)),
        RandomTranslation(0.08, 0.08),
        RandomRotation(0.05),
        RandomZoom(0.1),
        RandomTranslation(0.08, 0.08),
        RandomRotation(0.05),
        RandomZoom(0.1),
    ]
)

# Define the model
model = Sequential([
    Input((img_height, img_width, 3)),
    data_augmentation,
    Rescaling(1./255),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
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
    epochs=epochs
)

# Save the model
now = datetime.datetime.now()
model.save(f'models/classifier_{now.strftime("%d-%m-%Y_%H-%M")}.keras')
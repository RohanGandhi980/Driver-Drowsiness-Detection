import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "data/processed/mouth"   #contains yawn and no_yawn
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

os.makedirs("models", exist_ok=True)

# Using a single directory with a validation split to avoid needing split scripts
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='training', shuffle=True)

val = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='validation', shuffle=False)

#creating a light model
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train, validation_data=val, epochs=EPOCHS)

model.save("models/yawn_cnn.h5")
print("✅ Yawn CNN saved successfully!")
print("Class indices:", train.class_indices)

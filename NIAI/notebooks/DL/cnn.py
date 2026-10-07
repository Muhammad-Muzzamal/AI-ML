# Cat vs Dog CNN - v2 (new dataset source, Google URL was returning 403)
# Dataset: Microsoft "Cats and Dogs" (PetImages, ~25,000 images, ~800 MB)

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (128, 128)
BATCH = 32
EPOCHS = 10

# ---------------------------------------------------------------
# 1. Get the dataset
# ---------------------------------------------------------------
# OPTION A (automatic download):
URL = ("https://download.microsoft.com/download/3/E/1/"
       "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

# OPTION B (manual): if Option A fails, download "Dogs vs Cats" zip from
# Kaggle / Microsoft in your browser, extract it, and set DATA_DIR to the
# folder that CONTAINS the "Cat" and "Dog" folders, e.g.:
# DATA_DIR = r"D:\AI-ML\NIAI\datasets\PetImages"
DATA_DIR = None

if DATA_DIR is None:
    zip_path = tf.keras.utils.get_file(
        "kagglecatsanddogs.zip", origin=URL, extract=True)
    DATA_DIR = os.path.join(os.path.dirname(zip_path), "PetImages")

print("Using data from:", DATA_DIR)

# ---------------------------------------------------------------
# 2. Remove corrupted images (this dataset has some broken files)
# ---------------------------------------------------------------
removed = 0
for folder in ("Cat", "Dog"):
    folder_path = os.path.join(DATA_DIR, folder)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "rb") as f:
                is_jfif = b"JFIF" in f.peek(10)
        except Exception:
            is_jfif = False
        if not is_jfif:
            os.remove(fpath)
            removed += 1
print("Removed corrupted images:", removed)

# ---------------------------------------------------------------
# 3. Load data (80% train / 20% validation)
# ---------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary")
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary")
print("Classes:", train_ds.class_names)   # ['Cat', 'Dog'] -> Cat=0, Dog=1

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------------
# 4. Build the CNN
# ---------------------------------------------------------------
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    augment,
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ---------------------------------------------------------------
# 5. Train + save
# ---------------------------------------------------------------
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save("cat_dog_model.keras")

# ---------------------------------------------------------------
# 6. Predict on your own image
# ---------------------------------------------------------------
def predict_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
    prob = float(model.predict(arr)[0][0])
    label = "Dog" if prob > 0.5 else "Cat"
    conf = prob if prob > 0.5 else 1 - prob
    print(f"{label} ({conf * 100:.1f}% sure)")

# predict_image(r"D:\path\to\my_pet.jpg")
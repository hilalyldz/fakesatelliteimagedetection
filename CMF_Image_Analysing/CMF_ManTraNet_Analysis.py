#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model

# -------------------------------
# CONFIG
# -------------------------------
IMAGE_PATH = r"C:\Users\yild_hi\Desktop\Datasets\Real_RSCM_224\Real_RSCM_224\image\30.png"
WEIGHTS_PATH = "ManTraNet_Ptrain4.h5"

# -------------------------------
# MANTRANET ARCHITECTURE
# -------------------------------
def build_mantranet(input_shape=(None, None, 3)):
    inp = Input(shape=input_shape)

    x = Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = Conv2D(32, 3, padding="same", activation="relu")(x)
    x = Conv2D(64, 3, padding="same", activation="relu")(x)
    x = Conv2D(128, 3, padding="same", activation="relu")(x)
    x = Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    return model

# -------------------------------
# LOAD IMAGE
# -------------------------------
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

image_norm = image.astype(np.float32) / 255.0
image_input = np.expand_dims(image_norm, axis=0)

# -------------------------------
# LOAD MODEL + WEIGHTS
# -------------------------------
print("[INFO] Building ManTraNet architecture...")
model = build_mantranet()

print("[INFO] Loading pretrained weights...")
model.load_weights(WEIGHTS_PATH)

# -------------------------------
# PREDICTION
# -------------------------------
print("[INFO] Running Local Noise Consistency analysis...")
heatmap = model.predict(image_input)[0, :, :, 0]

heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

# -------------------------------
# VISUALIZATION
# -------------------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("ManTraNet Noise Inconsistency Map")
plt.imshow(heatmap, cmap="jet")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(image)
plt.imshow(heatmap, cmap="jet", alpha=0.5)
plt.axis("off")

# -------------------------------
# SAVE RESULT
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "ManTraNet_Image_Analysing")
os.makedirs(output_dir, exist_ok=True)

image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
output_path = os.path.join(output_dir, f"{image_name}_ManTraNet_LNC.png")

plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"[INFO] Saved to: {output_path}")

plt.show()

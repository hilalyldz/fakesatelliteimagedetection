#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from noiseprint import NoisePrintModel

# -------------------------------
# CONFIG
# -------------------------------
IMAGE_PATH = r"C:\Users\yild_hi\Desktop\Datasets\Real_RSCM_224\Real_RSCM_224\image\30.png"

# -------------------------------
# LOAD IMAGE
# -------------------------------
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

# Normalize for model input
image_norm = image.astype(np.float32) / 255.0
image_tensor = torch.from_numpy(image_norm.transpose(2,0,1)).unsqueeze(0)  # (1,C,H,W)

# -------------------------------
# LOAD NOISEPRINT MODEL
# -------------------------------
print("[INFO] Loading NoisePrint model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NoisePrintModel().to(device)
model.eval()

image_tensor = image_tensor.to(device)

# -------------------------------
# RUN LOCAL NOISE CONSISTENCY
# -------------------------------
print("[INFO] Running NoisePrint analysis...")
with torch.no_grad():
    noise_residual = model(image_tensor)[0].cpu().numpy()  # (H,W)

# Normalize heatmap for visualization
heatmap = (noise_residual - noise_residual.min()) / (noise_residual.max() - noise_residual.min() + 1e-8)

# Resize heatmap to original image
heatmap_resized = cv2.resize(heatmap, (w,h))

# -------------------------------
# VISUALIZATION
# -------------------------------
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("NoisePrint LNC Map")
plt.imshow(heatmap_resized, cmap="jet")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(image)
plt.imshow(heatmap_resized, cmap="jet", alpha=0.5)
plt.axis("off")

# -------------------------------
# SAVE RESULT
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "NoisePrint_Image_Analysing")
os.makedirs(output_dir, exist_ok=True)

image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
output_path = os.path.join(output_dir, f"{image_name}_NoisePrint_LNC.png")

plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"[INFO] Saved to: {output_path}")

plt.show()

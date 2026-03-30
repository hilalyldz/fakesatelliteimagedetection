#!/usr/bin/python
# -*- coding: utf-8 -*-
#===========================================================
#  File Name: CMF_Wavelet_Analysis.py
#  Author: Hilal Yildiz
#  Creation Date: 16-12-2025
#  Last Modified: 16-12-2025
#
#  Usage: python CMF_Wavelet_Analysis.py
#  Description: Analyze copy-move forgery (CMF) images using
#               Discrete Wavelet Transform (DWT). The script
#               decomposes an RGB image into approximation (LL)
#               and detail (LH, HL, HH) coefficients per channel,
#               visualizing the approximation to highlight main
#               image structures and potential duplicated regions.
#
#  Purpose:
#  1. Transform the image from spatial domain to frequency domain
#     while retaining spatial localization.
#  2. Highlight main patterns using low-frequency (LL) coefficients.
#  3. Facilitate block-based comparison for copy-move forgery detection.
#
#  Notes:
#  - LL coefficients capture most of the image structure.
#  - LH, HL, HH detail coefficients capture edges and fine textures.
#  - Wavelet transform is more sensitive to local changes than DCT.
#  - Visualization helps in detecting duplicated blocks visually.
#
#  Copyright (C) 2025 Hilal Yildiz
#  All rights reserved.
#
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================
"""
========================================================================
Copy-Move Forgery Analysis using Discrete Wavelet Transform (DWT)
========================================================================

Description:
------------
This script is designed to analyze images for potential copy-move forgery
(CMF) using the Discrete Wavelet Transform (DWT). The image is decomposed
into different frequency bands (sub-bands) to highlight structural and
textural information.

What is Wavelet Transform?
--------------------------
- Wavelet Transform is a mathematical technique that decomposes a signal
  (or image) into components at different scales (frequencies) and positions.
- In 2D images, DWT splits the image into four sub-bands:
    1. **LL** - Low-frequency approximation (main image structure)
    2. **LH** - Horizontal high-frequency details (edges/textures)
    3. **HL** - Vertical high-frequency details
    4. **HH** - Diagonal high-frequency details
- Unlike DCT, wavelets provide **both spatial and frequency localization**,
  which is especially useful for detecting local manipulations such as
  copy-move forgeries.

Why DWT for CMF?
----------------
1. **Local Analysis:** Wavelets capture local changes in an image, making
   it easier to detect small duplicated regions.
2. **Energy Compaction in LL:** Most image information is stored in LL,
   which can be compared across blocks to find copied areas.
3. **Edge Preservation:** High-frequency sub-bands (LH, HL, HH) preserve
   edge and texture details, helpful for more advanced analysis.

What the Results Mean:
---------------------
- **LL coefficients** represent the low-frequency content (general shapes
  and main structures).
- **LH, HL, HH coefficients** represent high-frequency content (edges,
  fine details, textures).
- Visualizing LL allows you to see the main patterns, and block-based
  comparison of LL can highlight duplicated regions.
- DWT is more sensitive to small local changes than DCT, making it useful
  when the copied region is slightly altered or textured differently.

Usage:
------
1. Load an RGB image.
2. Apply 2D DWT to each color channel.
3. Visualize the approximation (LL) sub-band or use it for block-based
   CMF detection.

========================================================================
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os

# Load the image
image_path = r"C:\Users\yild_hi\Desktop\Datasets\Real_RSCM_224\Real_RSCM_224\image\25.png"  # Replace with your image path
image = cv2.imread(image_path)
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert BGR to RGB

# Function to apply 2D DWT
def apply_dwt(channel, wavelet='haar'):
    coeffs2 = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

# Apply DWT to each channel
LL_r, LH_r, HL_r, HH_r = apply_dwt(image[:,:,0])
LL_g, LH_g, HL_g, HH_g = apply_dwt(image[:,:,1])
LL_b, LH_b, HL_b, HH_b = apply_dwt(image[:,:,2])

# Merge LL channels for visualization (approximation)
LL_image = np.stack([LL_r, LL_g, LL_b], axis=2)
LH_image = np.stack([LH_r, LH_g, LH_b], axis=2)
HL_image = np.stack([HL_r, HL_g, HL_b], axis=2)
HH_image = np.stack([HH_r, HH_g, HH_b], axis=2)

# Plot original and DWT approximation
plt.figure(figsize=(12,6))

plt.subplot(1,5,1)
plt.title("Original Image")
plt.imshow(image1)
plt.axis('off')

plt.subplot(1,5,2)
plt.title("DWT (LL)")
plt.imshow(LL_image.astype(np.uint8))
plt.axis('off')

plt.subplot(1,5,3)
plt.title("DWT (LH)")
plt.imshow(LH_image.astype(np.uint8))
plt.axis('off')

plt.subplot(1,5,4)
plt.title("DWT (HL)")
plt.imshow(HL_image.astype(np.uint8))
plt.axis('off')

plt.subplot(1,5,5)
plt.title("DWT (HH)")
plt.imshow(HH_image.astype(np.uint8))
plt.axis('off')

# -------- SAVE RESULT IN SCRIPT DIRECTORY (WAVELET) --------

# Script'in bulunduğu dizin
script_dir = os.path.dirname(os.path.abspath(__file__))

# Yeni klasör yolu
output_dir = os.path.join(script_dir, "Wavelet_Image_Analysing")
os.makedirs(output_dir, exist_ok=True)

# Çıktı dosya adı
image_name = os.path.splitext(os.path.basename(image_path))[0]
output_path = os.path.join(output_dir, f"{image_name}_DWT.png")

# Figürü kaydet
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[INFO] Wavelet analysis saved to: {output_path}")

plt.show()


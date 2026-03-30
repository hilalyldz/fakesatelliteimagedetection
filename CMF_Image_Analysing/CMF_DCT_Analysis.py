#!/usr/bin/python
# -*- coding: utf-8 -*-
#===========================================================
#  File Name: CMF_DCT_Analysis.py
#  Author: Hilal Yildiz
#  Creation Date: 16-12-2025
#  Last Modified: 16-12-2025
#
#  Usage: python CMF_DCT_Analysis.py
#  Description: Analyze copy-move forgery (CMF) images using
#               Discrete Cosine Transform (DCT). The script
#               converts an RGB image to the frequency domain,
#               computes 2D DCT per channel, and visualizes
#               the results. This helps in detecting potential
#               duplicated regions in the image.
#
#  Purpose:
#  1. Transform the image from spatial domain to frequency domain.
#  2. Highlight main patterns using low-frequency DCT coefficients.
#  3. Facilitate block-based comparison for copy-move forgery detection.
#
#  Notes:
#  - High DCT magnitude in low-frequency coefficients corresponds
#    to strong image structures.
#  - Visualization uses logarithmic scaling for better visibility.
#  - Can be extended to automatic detection of duplicated blocks.
#
#  Copyright (C) 2025 Hilal Yildiz
#  All rights reserved.
#
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================
"""
========================================================================
Copy-Move Forgery Analysis using Discrete Cosine Transform (DCT)
========================================================================

Description:
------------
This script is designed to analyze images for potential copy-move forgery
(CMF). The main idea is to transform the image from the spatial domain
(RGB pixel values) to the frequency domain using the Discrete Cosine
Transform (DCT).

Why DCT?
---------
1. **Energy Compaction:** DCT concentrates most of the image's information
   (energy) in a few low-frequency coefficients, which makes it easier
   to detect duplicated blocks.
2. **Block Comparison:** By applying DCT to overlapping blocks, we can
   efficiently compare regions for similarity, which is key in CMF detection.
3. **Robustness:** DCT coefficients are less sensitive to small changes
   like brightness or contrast adjustments, improving the detection of
   copied regions even after simple post-processing.

What the Results Mean:
---------------------
- The resulting DCT coefficients represent the frequency content of each
  image channel (R, G, B).
- High values in the low-frequency coefficients indicate strong patterns
  or structures in the image, while high-frequency coefficients capture
  edges and details.
- Visualizing the magnitude of DCT coefficients helps to understand which
  areas carry the main image information and may reveal duplicated
  regions when compared block by block.

Usage:
------
1. Load an RGB image.
2. Apply 2D DCT to each channel.
3. Visualize the DCT magnitudes or use them in a block-based CMF
   detection pipeline.

========================================================================
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import os

# Load the image
image_path = r"C:\Users\yild_hi\Desktop\Datasets\Real_RSCM_224\Real_RSCM_224\original\25.png"  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Function to apply 2D DCT
def apply_dct(channel):
    # Convert to float32 for DCT
    channel = np.float32(channel)
    # Apply DCT on rows
    dct_rows = dct(channel, axis=0, norm='ortho')
    # Apply DCT on columns
    dct_cols = dct(dct_rows, axis=1, norm='ortho')
    return dct_cols

# Apply DCT to each channel
dct_r = apply_dct(image[:,:,0])
dct_g = apply_dct(image[:,:,1])
dct_b = apply_dct(image[:,:,2])

# Merge channels back for visualization
dct_image = np.stack([np.abs(dct_r), np.abs(dct_g), np.abs(dct_b)], axis=2)

# Plot original and DCT images
# Plot original and DCT images
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1,2,2)
plt.title("DCT Magnitude (RGB Channels)")
plt.imshow(np.log1p(dct_image))
plt.axis('off')

# -------- SAVE RESULT IN SCRIPT DIRECTORY --------
# Get directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create output filename
image_name = os.path.splitext(os.path.basename(image_path))[0]
output_path = os.path.join(script_dir, f"{image_name}_DCT_analysis.png")

# Save figure
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[INFO] DCT analysis saved to: {output_path}")

plt.show()


#!/usr/bin/python
# -*- coding: utf-8 -*-
#===========================================================
#  File Name: CMF_Keypoint_Analysis.py
#  Author: Hilal Yildiz
#  Creation Date: 16-12-2025
#  Last Modified: 16-12-2025
#
#  Usage: python CMF_Keypoint_Analysis.py
#  Description: Analyze copy-move forgery (CMF) images using
#               keypoint-based feature detection methods such as
#               SIFT, ORB, or AKAZE. The script detects distinctive
#               keypoints and descriptors in the image, matches
#               them to find duplicated regions, and visualizes
#               potential forgeries.
#
#  Purpose:
#  1. Detect keypoints that are invariant to rotation, scale,
#     and slight illumination changes.
#  2. Compute feature descriptors for each keypoint.
#  3. Match keypoints within the same image to identify
#     potential duplicated (copy-move) regions.
#  4. Facilitate visualization of potential forged areas.
#
#  Notes:
#  - SIFT provides high accuracy for rotated/scaled duplications
#    but requires `opencv-contrib-python` (free since 2020).
#  - ORB and AKAZE are fully open-source, fast, and robust alternatives.
#  - Good matches are filtered using distance ratio tests to reduce
#    false positives.
#  - Can be combined with DCT or Wavelet analysis for hybrid CMF detection.
#
#  What the Results Mean:
#  - Detected keypoints indicate salient image features.
#  - Matched keypoints suggest duplicated regions (potential forgery).
#  - Visualization shows candidate regions for further analysis.
#
#  Copyright (C) 2025 Hilal Yildiz
#  All rights reserved.
#
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = "your_image.jpg"  # Replace with your image path
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect SIFT keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(
    gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(10, 6))
plt.imshow(image_with_keypoints, cmap='gray')
plt.title("SIFT Keypoints")
plt.axis('off')
plt.show()

# -----------------------------
# Optional: Matching keypoints for CMF detection
# -----------------------------
# Create a brute-force matcher
bf = cv2.BFMatcher()

# Match descriptors with themselves to find duplicated regions
matches = bf.knnMatch(descriptors, descriptors, k=2)

# Apply ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        # Ignore self-matches (keypoints matching themselves)
        if m.queryIdx != m.trainIdx:
            good_matches.append(m)

# Draw matches
match_img = cv2.drawMatches(
    gray, keypoints, gray, keypoints, good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(12, 8))
plt.imshow(match_img, cmap='gray')
plt.title("Potential Duplicated Regions (SIFT Matches)")
plt.axis('off')
plt.show()

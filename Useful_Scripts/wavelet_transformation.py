import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# --- Load image (in color) ---
image_path = r"C:\Users\yild_hi\Desktop\Datasets\Real_RSCM_224\RS_TQA_224\original\0.png"  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB

# --- Choose wavelet ---
wavelet = 'haar'  # You can also try 'db1', 'sym2', etc.

# --- Apply DWT separately to each RGB channel ---
coeffs_r = pywt.dwt2(image[:, :, 0], wavelet)
coeffs_g = pywt.dwt2(image[:, :, 1], wavelet)
coeffs_b = pywt.dwt2(image[:, :, 2], wavelet)

# Unpack coefficients
LL_r, (LH_r, HL_r, HH_r) = coeffs_r
LL_g, (LH_g, HL_g, HH_g) = coeffs_g
LL_b, (LH_b, HL_b, HH_b) = coeffs_b

# --- Stack each sub-band into RGB images ---
def merge_rgb(r, g, b):
    return np.stack([r, g, b], axis=-1).astype(np.uint8)

LL = merge_rgb(LL_r, LL_g, LL_b)
LH = merge_rgb(LH_r, LH_g, LH_b)
HL = merge_rgb(HL_r, HL_g, HL_b)
HH = merge_rgb(HH_r, HH_g, HH_b)

# --- Display the results ---
titles = ['Approximation (LL)', 'Horizontal Detail (LH)',
          'Vertical Detail (HL)', 'Diagonal Detail (HH)']
components = [LL, LH, HL, HH]

plt.figure(figsize=(10, 8))
for i, comp in enumerate(components):
    plt.subplot(2, 2, i + 1)
    plt.imshow(np.clip(comp, 0, 255).astype(np.uint8))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
